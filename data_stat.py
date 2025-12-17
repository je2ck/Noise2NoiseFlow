import tifffile
import numpy as np

def stats_with_percentile(path, percentiles=[0.1, 1, 5, 50, 95, 99, 99.9]):
    """
    Multi-page TIFF에 대해 histogram 기반으로 percentile까지 계산.
    0~65535 범위 uint16용 (원자 이미징용으로 최적화)
    """
    hist = np.zeros(65536, dtype=np.int64)
    total_pixels = 0

    try:
        with tifffile.TiffFile(path) as tif:
            for page in tif.pages:
                arr = page.asarray()

                # 1) uint16 아닐 경우 변환
                if arr.dtype != np.uint16:
                     arr = arr.astype(np.uint16)

                # 2) histogram 누적
                h, _ = np.histogram(arr, bins=65536, range=(0, 65536))
                hist += h
                total_pixels += arr.size
    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return None

    if total_pixels == 0:
        print("Error: No pixels found in the image.")
        return None

    # 기본 통계
    values = np.arange(65536)
    
    # 히스토그램에서 값이 존재하는 인덱스 찾기
    nonzero_indices = np.nonzero(hist)[0]
    if len(nonzero_indices) > 0:
        min_val = int(values[nonzero_indices[0]])
        max_val = int(values[nonzero_indices[-1]])
    else:
        min_val = 0
        max_val = 0

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
if __name__ == "__main__":
    file_path = "/Users/jaeickbae/Documents/research/denoise/data-prep/data/20251216_4_4_20_array/raw/4ms.tif"
    print(f"Analyzing: {file_path}")
    stats = stats_with_percentile(file_path)
    
    if stats:
        print("\n=== Statistics Result ===")
        print(f"Min: {stats['min']}")
        print(f"Max: {stats['max']}")
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Std: {stats['std']:.2f}")
        print("-" * 20)
        print("Percentiles:")
        # 보기 좋게 정렬해서 출력
        for p in sorted(stats["percentiles"].keys()):
            print(f"  {p:>5}% : {stats['percentiles'][p]}")