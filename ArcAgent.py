from collections import deque

import numpy as np

from ArcProblem import ArcProblem


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        test_input = arc_problem.test_set().get_input_data().data()
        predictions: list[np.ndarray] = []

        strategies = [
            self.solve_18419cfa_reflect_in_frames,
            self.solve_2546ccf6_mirror_richer_segment,
            self.solve_195ba7dc_or_halves,
            self.solve_81c0276b_frequency_histogram,
            self.solve_67c52801_pack_rectangles_into_slots,
            self.solve_60a26a3e_connect_crosses,
            self.solve_31d5ba1a_xor_halves,
            self._fill_closed_regions_with_hint_majority,
        ]

        for strategy in strategies:
            if not self._validate_on_training(strategy, arc_problem):
                continue
            candidate = strategy(test_input, arc_problem)
            if candidate is None:
                continue
            if not any(np.array_equal(candidate, existing) for existing in predictions):
                predictions.append(candidate)
            if len(predictions) >= 3:
                break

        if not predictions:
            predictions.append(test_input.copy())

        return predictions[:3]

    def _validate_on_training(self, strategy, arc_problem: ArcProblem) -> bool:
        """Use only strategies that exactly match every training example for this puzzle."""
        for train_set in arc_problem.training_set():
            train_in = train_set.get_input_data().data()
            expected = train_set.get_output_data().data()
            produced = strategy(train_in, arc_problem)
            if produced is None or not np.array_equal(produced, expected):
                return False
        return True




    def solve_195ba7dc_or_halves(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve 195ba7dc-like tasks:
        split by a single full-height separator column, then OR the two side masks
        and render result using output foreground color.
        """
        rows, cols = grid.shape
        sep_cols = [
            c
            for c in range(cols)
            if len(set(map(int, grid[:, c]))) == 1 and int(grid[0, c]) != 0 and c == (cols - 1 - c)
        ]
        if len(sep_cols) != 1:
            return None
        sep = sep_cols[0]
        left = grid[:, :sep]
        right = grid[:, sep + 1 :]
        if left.shape[1] != right.shape[1]:
            return None

        left_mask = left != 0
        right_mask = right != 0
        out_mask = np.logical_or(left_mask, right_mask)

        out_color = self._learn_nonzero_output_color(arc_problem)
        if out_color is None:
            return None

        out = np.zeros_like(left, dtype=int)
        out[out_mask] = out_color
        return out

    def solve_2546ccf6_mirror_richer_segment(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve 2546ccf6-like tasks:
        on grids split by full separator rows, for each non-separator color choose the row-segment
        where it appears most; mirror that segment vertically into other segments containing the same color
        with fewer pixels.
        """
        rows, cols = grid.shape

        full_row_colors = [int(grid[r, 0]) for r in range(rows) if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != 0]
        full_col_colors = [int(grid[0, c]) for c in range(cols) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != 0]
        sep_candidates = set(full_row_colors) & set(full_col_colors)
        if len(sep_candidates) != 1:
            return None
        sep = next(iter(sep_candidates))

        divider_rows = [r for r in range(rows) if np.all(grid[r, :] == sep)]
        segments = []
        prev = -1
        for dr in divider_rows + [rows]:
            r0, r1 = prev + 1, dr
            if r0 < r1:
                segments.append((r0, r1))
            prev = dr
        if not segments:
            return None

        out = grid.copy()
        colors = [int(c) for c in np.unique(grid) if int(c) not in (0, sep)]

        for color in colors:
            seg_counts = []
            for i, (r0, r1) in enumerate(segments):
                cnt = int(np.sum(grid[r0:r1, :] == color))
                if cnt > 0:
                    seg_counts.append((i, cnt))
            if len(seg_counts) < 2:
                continue

            src_i, src_cnt = max(seg_counts, key=lambda x: x[1])
            src_r0, src_r1 = segments[src_i]
            src_block = grid[src_r0:src_r1, :]
            src_flip = np.flipud(src_block)

            for tgt_i, tgt_cnt in seg_counts:
                if tgt_i == src_i or tgt_cnt >= src_cnt:
                    continue
                tgt_r0, tgt_r1 = segments[tgt_i]
                if (tgt_r1 - tgt_r0) != src_flip.shape[0]:
                    continue
                # copy only non-separator columns; keep separator columns intact
                for c in range(cols):
                    if np.all(grid[:, c] == sep):
                        continue
                    out[tgt_r0:tgt_r1, c] = src_flip[:, c]

        return out

    def solve_18419cfa_reflect_in_frames(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve 18419cfa-like tasks:
        for each connected frame component (most common non-zero color), reflect interior pattern color
        across frame bbox center (horizontal + vertical symmetry completion).
        """
        nz = [int(c) for c in np.unique(grid) if int(c) != 0]
        if len(nz) < 2:
            return None

        # infer frame color as the most frequent non-zero; fill color is the other one in training tasks
        counts = {c: int(np.sum(grid == c)) for c in nz}
        frame_color = max(counts.items(), key=lambda kv: kv[1])[0]
        fill_candidates = [c for c in nz if c != frame_color]
        if len(fill_candidates) != 1:
            return None
        fill_color = fill_candidates[0]

        rows, cols = grid.shape
        seen = np.zeros((rows, cols), dtype=bool)
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        out = grid.copy()

        for r in range(rows):
            for c in range(cols):
                if seen[r, c] or int(grid[r, c]) != frame_color:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and int(grid[nr, nc]) == frame_color:
                            seen[nr, nc] = True
                            q.append((nr, nc))

                if len(comp) < 10:
                    continue

                rmin = min(rr for rr, _ in comp)
                rmax = max(rr for rr, _ in comp)
                cmin = min(cc for _, cc in comp)
                cmax = max(cc for _, cc in comp)

                for rr in range(rmin, rmax + 1):
                    for cc in range(cmin, cmax + 1):
                        if int(grid[rr, cc]) != fill_color:
                            continue
                        rr_m = rmin + rmax - rr
                        cc_m = cmin + cmax - cc
                        for tr, tc in [(rr, cc), (rr_m, cc), (rr, cc_m), (rr_m, cc_m)]:
                            if rmin <= tr <= rmax and cmin <= tc <= cmax and int(out[tr, tc]) != frame_color:
                                out[tr, tc] = fill_color

        return out

    def solve_67c52801_pack_rectangles_into_slots(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve pattern like 67c52801:
        - Bottom row is a full "base" color.
        - Row above has base-colored separators that define horizontal slots (zero-runs).
        - Colored objects above are packed into those slots as filled rectangles, bottom-aligned
          to the separator row; rectangles may rotate to match slot width.
        """
        rows, cols = grid.shape
        if rows < 2:
            return None

        base_row = grid[rows - 1, :]
        if len(set(map(int, base_row))) != 1:
            return None
        base_color = int(base_row[0])

        slot_row = grid[rows - 2, :]
        slots: list[tuple[int, int]] = []
        c = 0
        while c < cols:
            if int(slot_row[c]) == 0:
                start = c
                while c < cols and int(slot_row[c]) == 0:
                    c += 1
                slots.append((start, c - 1))
            else:
                c += 1

        if not slots:
            return None

        # Find connected non-base objects in rows above the slot row.
        work = grid[: rows - 2, :]
        seen = np.zeros(work.shape, dtype=bool)
        objects: list[tuple[int, int, int]] = []  # (color, area, max_height)
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for r in range(work.shape[0]):
            for cc in range(work.shape[1]):
                val = int(work[r, cc])
                if val == 0 or val == base_color or seen[r, cc]:
                    continue
                q = deque([(r, cc)])
                seen[r, cc] = True
                cells = []
                while q:
                    cr, ccc = q.popleft()
                    cells.append((cr, ccc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, ccc + dc
                        if 0 <= nr < work.shape[0] and 0 <= nc < work.shape[1] and not seen[nr, nc] and int(work[nr, nc]) == val:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                area = len(cells)
                rmin = min(r0 for r0, _ in cells)
                rmax = max(r0 for r0, _ in cells)
                cmin = min(c0 for _, c0 in cells)
                cmax = max(c0 for _, c0 in cells)
                h = rmax - rmin + 1
                w = cmax - cmin + 1
                # store both dims in max_height placeholder as encoded tuple via list append below
                objects.append((val, area, h * 1000 + w))

        if len(objects) != len(slots):
            return None

        # Sort by area (small to large) to match slot order in training behavior.
        objects.sort(key=lambda x: (x[1], x[0]))

        out = np.zeros_like(grid)
        out[rows - 1, :] = base_color
        out[rows - 2, :] = slot_row

        for (color, area, enc_hw), (start, end) in zip(objects, slots):
            h0, w0 = divmod(enc_hw, 1000)
            slot_w = end - start + 1

            candidates = []
            for w in {w0, h0}:
                if w > 0 and area % w == 0:
                    h = area // w
                    candidates.append((w, h))

            picked = None
            for w, h in candidates:
                if w == slot_w and h <= rows - 1:
                    picked = (w, h)
                    break
            if picked is None:
                return None

            w, h = picked
            left = start
            top = (rows - 2) - (h - 1)
            bottom = rows - 2
            out[top : bottom + 1, left : left + w] = color

        return out

    def solve_81c0276b_frequency_histogram(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve pattern like 81c0276b:
        - Detect separator color that forms full divider rows and columns.
        - Partition into cell blocks between dividers.
        - Count non-separator colors appearing in cells.
        - Build compact histogram: one row per color, sorted by increasing frequency,
          each row filled left-to-right with that color repeated `count` times.
        """
        rows, cols = grid.shape

        full_row_colors = [int(grid[r, 0]) for r in range(rows) if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != 0]
        full_col_colors = [int(grid[0, c]) for c in range(cols) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != 0]
        sep_candidates = set(full_row_colors) & set(full_col_colors)
        if len(sep_candidates) != 1:
            return None
        sep = next(iter(sep_candidates))

        divider_rows = [r for r in range(rows) if np.all(grid[r, :] == sep)]
        divider_cols = [c for c in range(cols) if np.all(grid[:, c] == sep)]

        row_cuts = [-1] + divider_rows + [rows]
        col_cuts = [-1] + divider_cols + [cols]

        counts: dict[int, int] = {}
        for ri in range(len(row_cuts) - 1):
            r0, r1 = row_cuts[ri] + 1, row_cuts[ri + 1]
            if r0 >= r1:
                continue
            for ci in range(len(col_cuts) - 1):
                c0, c1 = col_cuts[ci] + 1, col_cuts[ci + 1]
                if c0 >= c1:
                    continue
                cell = grid[r0:r1, c0:c1]
                vals = [int(v) for v in np.unique(cell) if int(v) != 0 and int(v) != sep]
                if not vals:
                    continue
                if len(vals) != 1:
                    return None
                color = vals[0]
                counts[color] = counts.get(color, 0) + 1

        if not counts:
            return None

        items = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
        out_h = len(items)
        out_w = max(cnt for _, cnt in items)
        out = np.zeros((out_h, out_w), dtype=int)

        for r, (color, cnt) in enumerate(items):
            out[r, :cnt] = color

        return out

    def solve_60a26a3e_connect_crosses(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve pattern like 60a26a3e:
        - Detect "cross-with-empty-center" motifs made from one color (arms at up/down/left/right).
        - For motif centers that share the same row or column, draw a connector between the
          facing arms using the learned connector color.

        This reproduces the training behavior where red motifs are connected by blue lines
        only through the gap between motifs.
        """
        rows, cols = grid.shape
        nonzero = [int(c) for c in np.unique(grid) if int(c) != 0]
        if len(nonzero) != 1:
            return None
        motif_color = nonzero[0]

        line_color = self._learn_added_output_color(arc_problem)
        if line_color is None:
            return None

        centers: list[tuple[int, int]] = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if int(grid[r, c]) != 0:
                    continue
                if (
                    int(grid[r - 1, c]) == motif_color
                    and int(grid[r + 1, c]) == motif_color
                    and int(grid[r, c - 1]) == motif_color
                    and int(grid[r, c + 1]) == motif_color
                ):
                    centers.append((r, c))

        if not centers:
            return None

        out = grid.copy()

        # Horizontal connectors between motifs on the same row.
        by_row: dict[int, list[int]] = {}
        for r, c in centers:
            by_row.setdefault(r, []).append(c)
        for r, cols_on_row in by_row.items():
            cols_on_row.sort()
            for i in range(len(cols_on_row) - 1):
                left_c = cols_on_row[i]
                right_c = cols_on_row[i + 1]
                for c in range(left_c + 2, right_c - 1):
                    if int(out[r, c]) == 0:
                        out[r, c] = line_color

        # Vertical connectors between motifs in the same column.
        by_col: dict[int, list[int]] = {}
        for r, c in centers:
            by_col.setdefault(c, []).append(r)
        for c, rows_on_col in by_col.items():
            rows_on_col.sort()
            for i in range(len(rows_on_col) - 1):
                top_r = rows_on_col[i]
                bottom_r = rows_on_col[i + 1]
                for r in range(top_r + 2, bottom_r - 1):
                    if int(out[r, c]) == 0:
                        out[r, c] = line_color

        return out

    def _learn_added_output_color(self, arc_problem: ArcProblem) -> int | None:
        """Learn the unique non-zero color introduced in outputs but absent in inputs."""
        learned = set()
        for train_set in arc_problem.training_set():
            train_in = train_set.get_input_data().data()
            train_out = train_set.get_output_data().data()
            in_colors = {int(c) for c in np.unique(train_in) if int(c) != 0}
            out_colors = {int(c) for c in np.unique(train_out) if int(c) != 0}
            added = out_colors - in_colors
            if len(added) != 1:
                return None
            learned |= added
        if len(learned) != 1:
            return None
        return next(iter(learned))

    def solve_31d5ba1a_xor_halves(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve pattern like 31d5ba1a:
        - Input is two equal-height blocks stacked vertically.
        - Treat non-zero cells in the top and bottom block as binary masks.
        - Output is the XOR of those masks using the learned output color.

        Why this works for 31d5ba1a:
        the top 3 rows (color 9) and bottom 3 rows (color 4) are occupancy masks;
        output marks locations where exactly one of the two masks has a filled cell.
        """
        rows, cols = grid.shape
        if rows % 2 != 0:
            return None

        half = rows // 2
        top = grid[:half, :]
        bottom = grid[half:, :]

        top_mask = top != 0
        bottom_mask = bottom != 0
        xor_mask = np.logical_xor(top_mask, bottom_mask)

        output_color = self._learn_nonzero_output_color(arc_problem)
        if output_color is None:
            return None

        out = np.zeros((half, cols), dtype=int)
        out[xor_mask] = output_color
        return out

    def _learn_nonzero_output_color(self, arc_problem: ArcProblem) -> int | None:
        """Infer the unique non-zero output color from training outputs."""
        colors = set()
        for train_set in arc_problem.training_set():
            out = train_set.get_output_data().data()
            for color in np.unique(out):
                color = int(color)
                if color != 0:
                    colors.add(color)
        if len(colors) != 1:
            return None
        return next(iter(colors))

    def _fill_closed_regions_with_hint_majority(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:
        """
        Strategy for tasks like 4b6b68e5:
        - find large connected single-color components that act as closed outlines
        - for each enclosed area, fill with the most frequent non-zero hint color inside
        - erase stray non-outline/hint pixels outside filled enclosed regions
        """
        rows, cols = grid.shape
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        def components_of_color(color: int) -> list[list[tuple[int, int]]]:
            seen = np.zeros((rows, cols), dtype=bool)
            comps: list[list[tuple[int, int]]] = []
            for r in range(rows):
                for c in range(cols):
                    if seen[r, c] or int(grid[r, c]) != color:
                        continue
                    q = deque([(r, c)])
                    seen[r, c] = True
                    comp: list[tuple[int, int]] = []
                    while q:
                        cr, cc = q.popleft()
                        comp.append((cr, cc))
                        for dr, dc in dirs:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and int(grid[nr, nc]) == color:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    comps.append(comp)
            return comps

        def enclosed_cells(boundary_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
            wall = np.zeros((rows, cols), dtype=bool)
            for r, c in boundary_cells:
                wall[r, c] = True

            outside = np.zeros((rows, cols), dtype=bool)
            q = deque()

            for r in range(rows):
                for c in range(cols):
                    if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and not wall[r, c]:
                        outside[r, c] = True
                        q.append((r, c))

            while q:
                r, c = q.popleft()
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not wall[nr, nc] and not outside[nr, nc]:
                        outside[nr, nc] = True
                        q.append((nr, nc))

            enclosed_list: list[tuple[int, int]] = []
            for r in range(rows):
                for c in range(cols):
                    if not wall[r, c] and not outside[r, c]:
                        enclosed_list.append((r, c))
            return enclosed_list

        output = grid.copy()

        # Keep true outlines and newly filled pixels; all other non-zero pixels are erased.
        keep = np.zeros((rows, cols), dtype=bool)

        for color in map(int, np.unique(grid)):
            if color == 0:
                continue
            for comp in components_of_color(color):
                # Keep all substantial components (outlines/objects).
                if len(comp) >= 6:
                    for r, c in comp:
                        keep[r, c] = True
                else:
                    # tiny blobs are usually hints/noise
                    continue

                inside = enclosed_cells(comp)
                if not inside:
                    continue

                counts: dict[int, int] = {}
                for r, c in inside:
                    v = int(grid[r, c])
                    if v != 0:
                        counts[v] = counts.get(v, 0) + 1

                if not counts:
                    continue

                fill_color = max(counts.items(), key=lambda kv: kv[1])[0]
                for r, c in inside:
                    output[r, c] = fill_color
                    keep[r, c] = True

        for r in range(rows):
            for c in range(cols):
                if int(output[r, c]) != 0 and not keep[r, c]:
                    output[r, c] = 0

        return output
