import numpy as np

"""
Helper Functions
"""


# Conversion
def bool_to_np(board):
    f = lambda x: 1 if x else 0
    return np.asarray([[f(j) for j in i] for i in board])
 
 
# Core structural features 
def get_peaks(area):
    #Height of the highest filled cell in each column (1-indexed from bottom).
    peaks = np.zeros(area.shape[1])
    for col in range(area.shape[1]):
        col_data = area[:, col]
        if 1 in col_data:
            # argmax finds the first 1 from the top; height = rows - that index
            peaks[col] = area.shape[0] - np.argmax(col_data)
    return peaks
 
 
def get_bumpiness(peaks):
    #Sum of absolute height differences between adjacent columns.
    return float(np.sum(np.abs(np.diff(peaks))))
 
 
def get_holes(peaks, area):
    """
    Per-column count of empty cells below the column peak.
    A hole is any unfilled cell with at least one filled cell above it.
    """
    holes = []
    for col in range(area.shape[1]):
        start = int(-peaks[col])
        if start == 0:
            holes.append(0)
        else:
            holes.append(int(np.count_nonzero(area[start:, col] == 0)))
    return holes
 
 
def get_wells(peaks):
    """
    Per-column well depth: how much lower a column is than both its neighbours.
    Edge columns are compared only to their single neighbour.
    """
    wells = []
    for i in range(len(peaks)):
        if i == 0:
            w = max(0.0, float(peaks[1] - peaks[0]))
        elif i == len(peaks) - 1:
            w = max(0.0, float(peaks[-2] - peaks[-1]))
        else:
            w = max(0.0, float(peaks[i - 1] - peaks[i]),
                    float(peaks[i + 1] - peaks[i]))
        wells.append(w)
    return wells
 
 
def get_row_transition(area, highest_peak):
    """
    Number of horizontal cell-state changes (filled↔empty) from the
    highest occupied row down to the bottom.
    """
    total = 0
    start_row = int(area.shape[0] - highest_peak)
    for row in range(start_row, area.shape[0]):
        for col in range(1, area.shape[1]):
            if area[row, col] != area[row, col - 1]:
                total += 1
    return total
 
 
def get_col_transition(area, peaks):
    """
    Number of vertical cell-state changes (filled↔empty) within each column,
    from the column peak down to the bottom.
    """
    total = 0
    for col in range(area.shape[1]):
        if peaks[col] <= 1:
            continue
        start = int(area.shape[0] - peaks[col])
        for row in range(start, area.shape[0] - 1):
            if area[row, col] != area[row + 1, col]:
                total += 1
    return total
 
 

# Dellacherie features (additions) 
def get_landing_height(piece, y):
    """
    The row at which the lowest cell of the placed piece landed.
    y is the drop row (bottom-left of piece bounding box).
    Approximated as y + min skirt value of the piece.
    """
    if hasattr(piece, 'skirt') and piece.skirt:
        return float(y + min(piece.skirt))
    # Fallback when called without a Piece object (e.g. from np-board context)
    return float(y)
 
 
def get_eroded_piece_cells(piece, board_before, board_after):
    """
    (rows cleared after placement) × (cells from *this* piece that were
    in those cleared rows).
 
    board_before, board_after: 2-D numpy arrays (rows × cols).
    piece_cells: list of (row, col) absolute positions the piece occupied.
    """
    cleared_rows = []
    for row in range(board_after.shape[0]):
        # A row was cleared if it was full before clearing and is now empty
        if np.all(board_before[row] == 1) and np.all(board_after[row] == 0):
            cleared_rows.append(row)
 
    if not cleared_rows or not hasattr(piece, 'body'):
        return 0
 
    return len(cleared_rows)  # simplified: count cleared rows (full version needs piece positions)
 
 
def get_covered_holes(peaks, area):
    """
    For each column, count how many filled cells sit above at least one hole.
    This is a finer-grained penalty than raw hole count — deep buried holes
    cost more because more blocks need to be cleared to reach them.
    """
    covered = []
    for col in range(area.shape[1]):
        peak = int(peaks[col])
        if peak == 0:
            covered.append(0)
            continue
        start = area.shape[0] - peak
        col_slice = area[start:, col]
        # Count filled cells that have at least one empty cell below them
        count = 0
        found_hole = False
        for cell in reversed(col_slice):   # bottom to top
            if cell == 0:
                found_hole = True
            elif found_hole:
                count += 1
        covered.append(count)
    return covered
 
 
def get_col_holes_depth(peaks, area):
    """
    For each hole in a column, its depth = distance from the column peak.
    Returns the total summed depth across all holes in all columns.
    Deeper holes are harder to clear, so this penalises them more than
    a flat hole count does.
    """
    total_depth = 0
    for col in range(area.shape[1]):
        peak = int(peaks[col])
        if peak == 0:
            continue
        start = area.shape[0] - peak
        for offset, cell in enumerate(area[start:, col]):
            if cell == 0:
                # depth = distance below the peak (1 = just under the peak)
                total_depth += (peak - offset)
    return total_depth