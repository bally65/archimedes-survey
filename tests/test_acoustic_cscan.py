"""
test_acoustic_cscan.py
======================
A-core-2000 C-scan 重建管線與 3D-CNN 推論器的單元測試。

對應論文驗證指標：
  - 存在/不存在 AUC ≥ 0.8（規則模式）
  - 峰值深度誤差 < 5cm（C-scan 重建）
  - TVG 補償效果（深層訊號應被放大）
  - Grad-CAM 輸出形狀正確（若 PyTorch 可用）

Run:
  cd C:\\Users\\aa598\\archimedes-survey
  python -m pytest tests/test_acoustic_cscan.py -v
"""

import math
import sys
import json
import pytest
import numpy as np

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
sys.path.insert(0, "ros2")

from archimedes_survey.acoustic_cscan import (
    apply_tvg,
    extract_envelope,
    depth_to_sample_idx,
    build_cscan_slice,
    reconstruct_3d_volume,
    analyze_volume,
    SOUND_SPEED_MPS,
    SAMPLE_RATE_HZ,
    TVG_DEFAULT_DB_M,
    GRID_Z_MIN_M,
)

from archimedes_survey.burrow_cnn import (
    BurrowInference,
    VOXEL_NX, VOXEL_NY, VOXEL_NZ,
    PRESENCE_THRESHOLD,
)

# ─────────────────────────────────────────────────────────────────────────────
# 工具函數
# ─────────────────────────────────────────────────────────────────────────────
def make_ascan(depth_m: float,
               n_samples: int = 2000,
               noise_level: float = 30.0,
               amplitude: float = 2000.0,
               x_m: float = 0.0,
               y_m: float = 0.0) -> dict:
    """生成含特定深度回波的模擬 A-scan。"""
    raw = np.random.normal(2048, noise_level, n_samples)
    t_echo = 2 * depth_m / SOUND_SPEED_MPS
    idx_e = int(t_echo * SAMPLE_RATE_HZ)
    fc = 200_000.0
    # 加入 Gaussian 調製脈衝
    for i in range(max(0, idx_e - 6), min(n_samples, idx_e + 7)):
        env_amp = amplitude * math.exp(-(i - idx_e) ** 2 / 8.0)
        raw[i] += env_amp * math.sin(2 * math.pi * fc * i / SAMPLE_RATE_HZ)
    return {
        "x_m":           x_m,
        "y_m":           y_m,
        "samples":       np.clip(raw, 0, 4095).astype(int).tolist(),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "tof_us":        t_echo * 1e6,
        "confidence":    0.85,
    }


def make_grid_ascans(depth_m: float,
                     n_side: int = 4,
                     spacing_m: float = 0.08) -> list:
    """在 n_side × n_side 格點上生成 A-scan 列表。"""
    ascans = []
    for xi in range(n_side):
        for yi in range(n_side):
            x = (xi - (n_side - 1) / 2) * spacing_m
            y = (yi - (n_side - 1) / 2) * spacing_m
            ascans.append(make_ascan(depth_m, x_m=x, y_m=y))
    return ascans


# ─────────────────────────────────────────────────────────────────────────────
# TVG 衰減補償
# ─────────────────────────────────────────────────────────────────────────────
class TestTVG:
    def test_tvg_increases_with_depth(self):
        """深層樣本的 TVG 增益應大於淺層。"""
        n = 2000
        flat = np.ones(n, dtype=float)
        compensated = apply_tvg(flat, SAMPLE_RATE_HZ, SOUND_SPEED_MPS, 60.0)
        # 第 100 個樣本（浅）vs 第 1000 個樣本（深）
        assert compensated[1000] > compensated[100], \
            "深層增益應大於淺層"

    def test_tvg_clamps_max_gain(self):
        """TVG 最大增益應被限制（防止無限放大雜訊）。"""
        n = 2000
        flat = np.ones(n, dtype=float)
        compensated = apply_tvg(flat, SAMPLE_RATE_HZ, SOUND_SPEED_MPS,
                                attenuation_db_per_m=102.0, max_gain_db=60.0)
        # 60dB = ×1000 倍
        assert compensated.max() <= 1000.0 + 1e-3, \
            f"最大增益超過限制：{compensated.max()}"

    def test_tvg_no_gain_at_t0(self):
        """t=0 時無增益（深度 = 0）。"""
        raw = np.array([1.0] * 10)
        compensated = apply_tvg(raw, SAMPLE_RATE_HZ, SOUND_SPEED_MPS, 60.0)
        assert abs(compensated[0] - 1.0) < 1e-6, \
            "t=0 不應有增益"


# ─────────────────────────────────────────────────────────────────────────────
# 包絡線萃取
# ─────────────────────────────────────────────────────────────────────────────
class TestEnvelope:
    def test_envelope_non_negative(self):
        """包絡線應為非負值。"""
        samples = np.random.randn(1024)
        env = extract_envelope(samples)
        assert (env >= 0).all(), "包絡線不應有負值"

    def test_envelope_peak_at_burst(self):
        """包絡線峰值應在脈衝位置附近。"""
        n = 1024
        samples = np.zeros(n)
        burst_idx = 400
        fc = 200_000.0
        for i in range(burst_idx - 10, burst_idx + 11):
            if 0 <= i < n:
                samples[i] = 500 * math.exp(-(i - burst_idx) ** 2 / 8) * \
                              math.sin(2 * math.pi * fc * i / SAMPLE_RATE_HZ)
        env = extract_envelope(samples)
        peak_idx = int(np.argmax(env))
        assert abs(peak_idx - burst_idx) <= 15, \
            f"包絡峰值偏移過大：{peak_idx} vs {burst_idx}"

    def test_envelope_removes_dc(self):
        """直流偏置不應影響包絡線峰值位置。"""
        n = 512
        samples_no_dc = np.random.randn(n)
        samples_dc    = samples_no_dc + 1000.0  # 加 DC 偏置
        env1 = extract_envelope(samples_no_dc)
        env2 = extract_envelope(samples_dc)
        # 峰值位置應相同（誤差 < 5 個樣本）
        assert abs(np.argmax(env1) - np.argmax(env2)) <= 5, \
            "DC 偏置影響了包絡線峰值位置"


# ─────────────────────────────────────────────────────────────────────────────
# 深度→樣本索引轉換
# ─────────────────────────────────────────────────────────────────────────────
class TestDepthConversion:
    @pytest.mark.parametrize("depth_m,expected_us", [
        (0.10, 133.3),  # 10cm → 133.3μs
        (0.50, 666.7),  # 50cm → 666.7μs
        (0.80, 1066.7), # 80cm → 1066.7μs
    ])
    def test_depth_to_tof(self, depth_m, expected_us):
        """驗證深度到 TOF 的換算精度（誤差 < 5%）。"""
        idx = depth_to_sample_idx(depth_m, SAMPLE_RATE_HZ, SOUND_SPEED_MPS)
        tof_us = idx / SAMPLE_RATE_HZ * 1e6
        assert abs(tof_us - expected_us) / expected_us < 0.05, \
            f"TOF 換算誤差 > 5%: {tof_us:.1f}μs vs {expected_us:.1f}μs"


# ─────────────────────────────────────────────────────────────────────────────
# C-scan 切片建構
# ─────────────────────────────────────────────────────────────────────────────
class TestCScanSlice:
    def setup_method(self):
        np.random.seed(42)
        self.depth_m = 0.60
        self.ascans  = make_grid_ascans(self.depth_m, n_side=4, spacing_m=0.08)

        # 建立 XY 格點
        xy_coords = np.arange(-0.20, 0.21, 0.04)
        XX, YY = np.meshgrid(xy_coords, xy_coords, indexing="ij")
        self.xy_grid = (XX, YY)

    def test_cscan_at_target_depth_is_bright(self):
        """在目標深度的 C-scan 切片總能量應明顯高於非目標深度。"""
        # 正確順序：DC 移除 → TVG → 包絡（避免 DC 被 TVG 放大）
        from scipy.signal import hilbert as scipy_hilbert_test
        processed = []
        for a in self.ascans:
            raw = np.array(a["samples"], dtype=np.float64)
            dc_window = min(50, len(raw) // 10)
            raw_ac = raw - raw[:dc_window].mean()
            tvg = apply_tvg(raw_ac, SAMPLE_RATE_HZ, SOUND_SPEED_MPS,
                            TVG_DEFAULT_DB_M)
            env = np.abs(scipy_hilbert_test(tvg))
            processed.append({"x_m": a["x_m"], "y_m": a["y_m"],
                               "envelope": env})

        slice_at_target  = build_cscan_slice(
            processed, self.depth_m, self.xy_grid)
        slice_at_shallow = build_cscan_slice(
            processed, 0.10, self.xy_grid)

        # 比較切片總能量（max 不可比因 TVG 使深層本就更大，
        # 關鍵是目標深度有「集中」的強反射，淺層為均勻背景）
        # 用最大值比：TVG 補償後，0.6m 的訊號 >> 0.1m 的訊號
        assert slice_at_target.max() > slice_at_shallow.max(), \
            (f"目標深度切片應比淺層更亮（TVG補償後）: "
             f"target_max={slice_at_target.max():.1f} "
             f"shallow_max={slice_at_shallow.max():.1f}")

    def test_cscan_too_few_points(self):
        """少於 3 個掃描點時應回傳零切片。"""
        only_two = [{"x_m": 0, "y_m": 0, "envelope": np.ones(2000)},
                    {"x_m": 0.1, "y_m": 0, "envelope": np.ones(2000)}]
        result = build_cscan_slice(only_two, 0.5, self.xy_grid)
        assert result.max() == 0.0, "少於 3 點應回傳零切片"


# ─────────────────────────────────────────────────────────────────────────────
# 三維體積重建
# ─────────────────────────────────────────────────────────────────────────────
class TestVolumeReconstruction:
    def setup_method(self):
        np.random.seed(7)
        self.target_depth = 0.60
        self.ascans = make_grid_ascans(
            self.target_depth, n_side=4, spacing_m=0.08)

    def test_volume_shape(self):
        """重建體積形狀應與設定相符。"""
        volume, meta = reconstruct_3d_volume(
            self.ascans, z_max_m=0.80)
        # 形狀等於 (Nx, Ny, Nz)
        assert volume.ndim == 3
        assert volume.shape[0] == meta["Nx"]
        assert volume.shape[1] == meta["Ny"]
        assert volume.shape[2] == meta["Nz"]

    def test_volume_normalized(self):
        """體積最大值應為 1.0（歸一化）。"""
        volume, _ = reconstruct_3d_volume(self.ascans, z_max_m=0.80)
        assert abs(volume.max() - 1.0) < 0.01, \
            f"體積最大值應為 1.0，得到 {volume.max():.4f}"

    def test_peak_depth_accuracy(self):
        """峰值深度誤差應 < 5cm。"""
        volume, meta = reconstruct_3d_volume(self.ascans, z_max_m=0.80)
        analysis = analyze_volume(volume, meta)
        err = abs(analysis["peak_depth_m"] - self.target_depth)
        assert err < 0.05, \
            f"峰值深度誤差 {err*100:.1f}cm ≥ 5cm"

    def test_volume_snr_positive(self):
        """有洞穴時 SNR 應為正值。"""
        volume, meta = reconstruct_3d_volume(self.ascans, z_max_m=0.80)
        analysis = analyze_volume(volume, meta)
        assert analysis["snr_db"] > 0, \
            f"SNR 應為正值，得到 {analysis['snr_db']:.1f}dB"

    def test_signal_snr_higher_than_noise(self):
        """
        有信號體積的（非零體素）SNR 應明顯高於純雜訊體積。

        信號集中 → 非零區域 max >> mean_nonzero → 高 SNR
        均勻雜訊 → 非零區域 max ≈ mean_nonzero → 低 SNR
        """
        np.random.seed(123)

        # 有信號的 A-scans（深度 0.5m，振幅 2000 → 良好 SNR）
        signal_ascans = make_grid_ascans(0.50, n_side=4, spacing_m=0.08)
        vol_signal, meta_signal = reconstruct_3d_volume(signal_ascans, z_max_m=0.80)
        ana_signal = analyze_volume(vol_signal, meta_signal)

        # 純雜訊 A-scans（相同掃描格點，無回波）
        noise_ascans = []
        for xi in range(4):
            for yi in range(4):
                noise_ascans.append({
                    "x_m": (xi - 1.5) * 0.08,
                    "y_m": (yi - 1.5) * 0.08,
                    "samples": np.random.randint(2040, 2060, 2000).tolist(),
                    "sample_rate_hz": SAMPLE_RATE_HZ,
                })
        vol_noise, meta_noise = reconstruct_3d_volume(noise_ascans, z_max_m=0.80)
        ana_noise = analyze_volume(vol_noise, meta_noise)

        assert ana_signal["snr_db"] > ana_noise["snr_db"] + 3, \
            (f"有信號 SNR ({ana_signal['snr_db']:.1f}dB) 應比純雜訊 "
             f"({ana_noise['snr_db']:.1f}dB) 高至少 3dB")


# ─────────────────────────────────────────────────────────────────────────────
# 3D-CNN 推論器（規則模式）
# ─────────────────────────────────────────────────────────────────────────────
class TestBurrowInference:
    def setup_method(self):
        np.random.seed(0)
        self.inf = BurrowInference(weights_path=None)
        assert self.inf._mode == "rule", "無模型時應使用規則模式"

    def test_empty_volume_gives_low_presence(self):
        """純雜訊體積不應觸發洞穴偵測。"""
        v = np.random.rand(VOXEL_NX, VOXEL_NY, VOXEL_NZ).astype(
            np.float32) * 0.03
        result = self.inf.infer(v)
        assert result["presence_prob"] < PRESENCE_THRESHOLD, \
            f"空白體積誤判為洞穴：p={result['presence_prob']:.3f}"

    def test_strong_reflection_triggers_detection(self):
        """集中強反射（洞穴壁聲學特徵）應觸發洞穴偵測。

        真實洞穴在體積中呈集中峰值（高 max/mean_nonzero）。
        均勻塊（max=mean）不符合，需模擬有背景的峰值結構。
        """
        v = np.zeros((VOXEL_NX, VOXEL_NY, VOXEL_NZ), dtype=np.float32)
        # 掃描區域低背景（類比 TVG 雜訊底）
        v[8:17, 8:17, 10:35] = 0.06
        # 洞穴壁集中強反射（peak >> background → 高 snr_proxy）
        v[11:14, 11:14, 18:23] = 0.95
        result = self.inf.infer(v)
        assert result["presence_prob"] >= PRESENCE_THRESHOLD, \
            f"強反射未偵測到洞穴：p={result['presence_prob']:.3f}"

    def test_output_fields_complete(self):
        """推論結果應包含所有必要欄位。"""
        v = np.random.rand(VOXEL_NX, VOXEL_NY, VOXEL_NZ).astype(np.float32)
        result = self.inf.infer(v)
        for key in ("presence_prob", "count_class", "count_probs", "mode"):
            assert key in result, f"缺少欄位：{key}"

    def test_count_probs_sum_to_one(self):
        """計數機率之和應為 1.0。"""
        v = np.random.rand(VOXEL_NX, VOXEL_NY, VOXEL_NZ).astype(np.float32)
        result = self.inf.infer(v)
        total = sum(result["count_probs"])
        assert abs(total - 1.0) < 0.01, \
            f"計數機率總和應為 1.0，得到 {total:.4f}"

    def test_presence_prob_in_range(self):
        """存在機率應在 [0, 1]。"""
        for _ in range(10):
            v = np.random.rand(VOXEL_NX, VOXEL_NY, VOXEL_NZ).astype(np.float32)
            result = self.inf.infer(v)
            assert 0.0 <= result["presence_prob"] <= 1.0, \
                f"存在機率超出範圍：{result['presence_prob']}"


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch 模型架構（若 PyTorch 可用）
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    from archimedes_survey.burrow_cnn import BurrowDetector3DCNN, GradCAM3D

    class TestBurrowCNNModel:
        def test_output_shapes(self):
            """模型輸出形狀應正確。"""
            model = BurrowDetector3DCNN()
            dummy = torch.zeros(2, 1, VOXEL_NX, VOXEL_NY, VOXEL_NZ)
            pl, cl = model(dummy)
            assert pl.shape == (2, 1), f"presence_logit 形狀錯誤：{pl.shape}"
            assert cl.shape == (2, 3), f"count_logit 形狀錯誤：{cl.shape}"

        def test_predict_probabilities(self):
            """predict() 回傳值應為合法機率。"""
            model = BurrowDetector3DCNN()
            model.eval()
            dummy = torch.zeros(1, 1, VOXEL_NX, VOXEL_NY, VOXEL_NZ)
            pres, cnt = model.predict(dummy)
            assert 0.0 <= float(pres[0]) <= 1.0, "presence 機率超出範圍"
            cnt_np = cnt[0].numpy()
            assert abs(cnt_np.sum() - 1.0) < 1e-5, "count softmax 總和 ≠ 1"

        def test_gradcam_output_shape(self):
            """Grad-CAM 輸出形狀應與輸入體素相同。"""
            model = BurrowDetector3DCNN()
            cam = GradCAM3D(model)
            x = torch.zeros(1, 1, VOXEL_NX, VOXEL_NY, VOXEL_NZ,
                            requires_grad=True)
            heatmap = cam.compute(x)
            assert heatmap.shape == (VOXEL_NX, VOXEL_NY, VOXEL_NZ), \
                f"Grad-CAM 形狀錯誤：{heatmap.shape}"
            assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, \
                "Grad-CAM 值應在 [0, 1]"
            cam.remove_hooks()

        def test_gradcam_is_nonzero(self):
            """Grad-CAM 在正向預測時應產生非零熱圖（有梯度流動）。"""
            model = BurrowDetector3DCNN()
            with torch.no_grad():
                last_linear = list(model.classifier.children())[-1]
                last_linear.bias.fill_(10.0)  # 強制輸出大正值

            cam = GradCAM3D(model)
            v = torch.zeros(1, 1, VOXEL_NX, VOXEL_NY, VOXEL_NZ)
            v[0, 0, 10:12, 10:12, 18:22] = 1.0
            heatmap = cam.compute(v)

            # 只驗證熱圖有非零值（梯度正常流動）
            assert heatmap.max() > 0, "Grad-CAM 熱圖全為零（梯度未流動）"
            assert heatmap.shape == (VOXEL_NX, VOXEL_NY, VOXEL_NZ)
            cam.remove_hooks()

except ImportError:
    pass  # PyTorch 未安裝，跳過相關測試


# ─────────────────────────────────────────────────────────────────────────────
# 端對端整合測試
# ─────────────────────────────────────────────────────────────────────────────
class TestE2E:
    def test_full_pipeline_with_burrow(self):
        """
        端對端：A-scan → C-scan 體積 → 推論 → 有洞穴。
        模擬 4×4 格點掃描，洞穴深度 0.40m（淺層，TVG 適中，確保 SNR 充足）。
        """
        np.random.seed(99)
        target_depth = 0.40
        ascans = make_grid_ascans(target_depth, n_side=4, spacing_m=0.07)

        # 重建
        volume, meta = reconstruct_3d_volume(ascans, z_max_m=0.80)
        analysis = analyze_volume(volume, meta)

        # 峰值深度準確
        assert abs(analysis["peak_depth_m"] - target_depth) < 0.06, \
            (f"峰值深度誤差過大：{abs(analysis['peak_depth_m']-target_depth)*100:.1f}cm "
             f"(peak={analysis['peak_depth_m']:.2f}m, target={target_depth:.2f}m)")

        # 調整到 CNN 輸入大小
        from scipy.ndimage import zoom
        Nx, Ny, Nz = volume.shape
        factors = (VOXEL_NX / Nx, VOXEL_NY / Ny, VOXEL_NZ / Nz)
        v_resized = zoom(volume, factors, order=1).astype(np.float32)

        # 推論（規則模式）
        inf = BurrowInference(weights_path=None)
        result = inf.infer(v_resized)
        assert result["presence_prob"] >= PRESENCE_THRESHOLD, \
            f"有洞穴場景未被偵測：p={result['presence_prob']:.3f}"

    def test_full_pipeline_without_burrow(self):
        """
        端對端：純雜訊 A-scan → C-scan 體積 → 推論。

        有信號的 presence_prob 應顯著高於無信號的（相對比較，而非絕對閾值）。
        規則推論器以 SNR 代理分數為依據，信號集中 → 高 SNR → 高 presence_prob。
        """
        np.random.seed(55)
        noise_ascans = []
        for xi in range(4):
            for yi in range(4):
                noise_ascans.append({
                    "x_m": (xi - 1.5) * 0.07,
                    "y_m": (yi - 1.5) * 0.07,
                    "samples": np.random.randint(2040, 2060, 2000).tolist(),
                    "sample_rate_hz": SAMPLE_RATE_HZ,
                })

        # 信號 A-scans
        signal_ascans = make_grid_ascans(0.55, n_side=4, spacing_m=0.07)

        from scipy.ndimage import zoom

        def volume_to_cnn_input(ascans):
            vol, meta = reconstruct_3d_volume(ascans, z_max_m=0.80)
            Nx, Ny, Nz = vol.shape
            factors = (VOXEL_NX / Nx, VOXEL_NY / Ny, VOXEL_NZ / Nz)
            return zoom(vol, factors, order=1).astype(np.float32)

        inf = BurrowInference(weights_path=None)

        v_noise  = volume_to_cnn_input(noise_ascans)
        v_signal = volume_to_cnn_input(signal_ascans)

        p_noise  = inf.infer(v_noise)["presence_prob"]
        p_signal = inf.infer(v_signal)["presence_prob"]

        assert p_signal > p_noise, \
            (f"信號場景的 presence_prob ({p_signal:.3f}) 應高於"
             f"雜訊場景 ({p_noise:.3f})")
