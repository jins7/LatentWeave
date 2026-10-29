def get_vision_resolution(dataset: str = "mmmu"):
    dataset = dataset.lower()
    if dataset == "mmmu":
        MIN_PIXELS = 1280 * 28 * 28
        MAX_PIXELS = 5120 * 28 * 28
    elif dataset == "mathvision":
        MIN_PIXELS = 768 * 28 * 28
        MAX_PIXELS = 5120 * 28 * 28
    else:
        print(f"Warning: unknown dataset '{dataset}', using MMMU defaults")
        MIN_PIXELS = 1280 * 28 * 28
        MAX_PIXELS = 5120 * 28 * 28
    return MIN_PIXELS, MAX_PIXELS


MMMU_MIN_PIXELS = 1280 * 28 * 28
MMMU_MAX_PIXELS = 5120 * 28 * 28
MATHVISION_MIN_PIXELS = 768 * 28 * 28
MATHVISION_MAX_PIXELS = 5120 * 28 * 28
