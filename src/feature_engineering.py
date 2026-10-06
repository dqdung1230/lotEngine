import numpy as np

def compute_gaps(history_array, max_num=35):
    num_draws = len(history_array)
    gaps = np.zeros((num_draws, max_num), dtype=np.int32)
    current_gap = np.zeros(max_num, dtype=np.int32)
    
    for t in range(num_draws):
        gaps[t] = current_gap.copy()
        current_gap += 1
        for num in history_array[t]:
            current_gap[num - 1] = 0
            
    return gaps

def compute_frequency(history_array, window=30, max_num=35):
    num_draws = len(history_array)
    freq = np.zeros((num_draws, max_num), dtype=np.float32)
    
    for t in range(window, num_draws):
        sub_history = history_array[t-window:t]
        counts = np.zeros(max_num)
        for draw in sub_history:
            for num in draw:
                counts[num - 1] += 1
        freq[t] = counts / window
        
    return freq
