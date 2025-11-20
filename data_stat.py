import tifffile
import numpy as np

def stats_with_percentile(path, percentiles=[1, 5, 50, 95, 99]):
    """
    Multi-page TIFF에 대해 histogram 기반으로 percentile까지 계산.
    0~65535 범위 uint16용 (원자 이미징용으로 최적화)
    """
    hist = np.zeros(65536, dtype=np.int64)
    total_pixels = 0

    with tifffile.TiffFile(path) as tif:
        for page in tif.pages:
            arr = page.asarray()

            # 1) uint16 아닐 경우 float로 올림
            arr = arr.astype(np.uint16, copy=False)

            # 2) histogram 누적
            h, _ = np.histogram(arr, bins=65536, range=(0, 65535))
            hist += h
            total_pixels += arr.size

    # 기본 통계
    values = np.arange(65536)
    min_val = int(values[np.nonzero(hist)[0][0]])
    max_val = int(values[np.nonzero(hist)[0][-1]])
    mean = float((values * hist).sum() / total_pixels)
    var = float((hist * (values - mean) ** 2).sum() / total_pixels)
    std = np.sqrt(var)

    # percentile 계산
    cdf = np.cumsum(hist)
    percentile_results = {}
    for p in percentiles:
        threshold = p / 100 * total_pixels
        idx = np.searchsorted(cdf, threshold)
        percentile_results[p] = int(idx)

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "std": std,
        "percentiles": percentile_results,
        "total_pixels": total_pixels,
    }

# 사용 예시
stats = stats_with_percentile("20251020_8ms_10ms_20ms_between_20ms_rep20000_64x64.tif")
print(stats)