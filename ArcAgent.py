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
            self.solve_c8b7cc0f_count_markers_inside_one_bbox,
            self.solve_992798f6_kinked_path_from_two_to_one,
            self.solve_e9b4f6fc_extract_and_recolor_panel_dynamic,
            self.solve_67c52801_pack_colors_into_support_slots_dynamic,
            self.solve_c48954c1_reflective_block_tiling,
            self.solve_c1990cce_generate_v_and_mod4_diagonals,
            self.solve_e98196ab_overlay_halves_across_divider,
            self.solve_dc433765_move_three_toward_four,
            self.solve_f35d900a_corner_blocks_with_dashed_connectors,
            self.solve_f8a8fe49_reflect_fives_across_twos,
            self.solve_cf98881b_overlay_three_panels_left_priority,
            self.solve_74dd1130_transpose_grid,
            self.solve_f2829549_nor_left_right_halves,
            self.solve_f25ffba3_reflect_lower_half_to_top,
            self.solve_d687bc17_route_edge_colors_to_inner_border,
            self.solve_f76d97a5_project_fives_to_other_color,
            self.solve_ce4f8723_or_top_bottom_masks,
            self.solve_ce22a75a_expand_points_to_3x3,
            self.solve_b94a9452_swap_inner_outer,
            self.solve_623ea044_draw_diagonals_through_point,
            self.solve_28e73c20_spiral_maze,
            self.solve_bbb1b8b6_overlay_if_disjoint,
            self.solve_992798f6_dominant_axis_path,
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

    def solve_c8b7cc0f_count_markers_inside_one_bbox(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve c8b7cc0f-like tasks dynamically:
        count marker-color cells inside the bounding box of color-1 structure, then render that
        count into a 3x3 row-major fill using the marker color.
        """
        ones = np.argwhere(grid == 1)
        if len(ones) == 0:
            return None

        marker_colors = [int(c) for c in np.unique(grid) if int(c) not in (0, 1)]
        if len(marker_colors) != 1:
            return None
        marker = marker_colors[0]

        rmin, cmin = ones.min(axis=0)
        rmax, cmax = ones.max(axis=0)
        rmin, cmin, rmax, cmax = map(int, (rmin, cmin, rmax, cmax))

        inside_count = int(np.sum(grid[rmin : rmax + 1, cmin : cmax + 1] == marker))
        if inside_count <= 0:
            return None

        out = np.zeros((3, 3), dtype=int)
        fill_count = min(inside_count, 9)
        idx = 0
        for r in range(3):
            for c in range(3):
                if idx < fill_count:
                    out[r, c] = marker
                idx += 1
        return out

    def solve_992798f6_kinked_path_from_two_to_one(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve 992798f6-like tasks dynamically:
        connect color-2 to color-1 with color-3 using a kinked monotone path:
          1) one diagonal step toward target,
          2) remaining dominant-axis extra steps,
          3) remaining diagonal steps.
        """
        pts1 = np.argwhere(grid == 1)
        pts2 = np.argwhere(grid == 2)
        if len(pts1) != 1 or len(pts2) != 1:
            return None

        r1, c1 = map(int, pts1[0])  # target
        r2, c2 = map(int, pts2[0])  # source
        dr = r1 - r2
        dc = c1 - c2
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
        abs_dr, abs_dc = abs(dr), abs(dc)
        diag_steps = min(abs_dr, abs_dc)
        extra = abs(abs_dr - abs_dc)
        vertical_dominant = abs_dr >= abs_dc

        out = np.zeros_like(grid)
        out[r1, c1] = 1
        out[r2, c2] = 2

        cr, cc = r2, c2

        # One diagonal step first (if possible).
        if diag_steps > 0:
            cr += step_r
            cc += step_c
            if (cr, cc) != (r1, c1):
                out[cr, cc] = 3
            diag_steps -= 1

        # Dominant-axis extra steps.
        for _ in range(extra):
            if vertical_dominant:
                cr += step_r
            else:
                cc += step_c
            if (cr, cc) != (r1, c1):
                out[cr, cc] = 3

        # Remaining diagonal steps.
        for _ in range(diag_steps):
            cr += step_r
            cc += step_c
            if (cr, cc) != (r1, c1):
                out[cr, cc] = 3

        return out

    def solve_e9b4f6fc_extract_and_recolor_panel_dynamic(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve e9b4f6fc-like tasks dynamically:
        - find the large solid rectangular panel (largest non-zero connected component),
        - crop it,
        - read external 2-cell horizontal mapping pairs (target, source),
        - recolor panel values by source->target mapping.
        """
        def get_connected_components(arr: np.ndarray) -> list[list[tuple[int, int]]]:
            rows, cols = arr.shape
            nz = np.argwhere(arr != 0)
            if len(nz) == 0:
                return []

            seen = np.zeros_like(arr, dtype=bool)
            components: list[list[tuple[int, int]]] = []
            for r0, c0 in nz:
                r0, c0 = int(r0), int(c0)
                if seen[r0, c0]:
                    continue
                q = deque([(r0, c0)])
                seen[r0, c0] = True
                comp: list[tuple[int, int]] = []
                while q:
                    r, c = q.popleft()
                    comp.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < rows and 0 <= cc < cols and not seen[rr, cc] and arr[rr, cc] != 0:
                            seen[rr, cc] = True
                            q.append((rr, cc))
                components.append(comp)
            return components

        def get_solid_panel_candidates(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
            candidates: list[tuple[int, int, int, int]] = []
            for comp in get_connected_components(arr):
                rs = [r for r, _ in comp]
                cs = [c for _, c in comp]
                rmin, rmax = min(rs), max(rs)
                cmin, cmax = min(cs), max(cs)
                h = rmax - rmin + 1
                w = cmax - cmin + 1
                if len(comp) == h * w:
                    candidates.append((rmin, rmax, cmin, cmax))
            candidates.sort(key=lambda box: (-(box[1] - box[0] + 1) * (box[3] - box[2] + 1), box[0], box[2]))
            return candidates

        def make_output_for_box(arr: np.ndarray, box: tuple[int, int, int, int], mode: str) -> np.ndarray | None:
            rows, cols = arr.shape
            rmin, rmax, cmin, cmax = box
            panel = arr[rmin : rmax + 1, cmin : cmax + 1].copy()
            panel_colors = {int(v) for v in np.unique(panel) if int(v) != 0}
            if not panel_colors:
                return None

            mapping: dict[int, int] = {}
            for r in range(rows):
                for c in range(cols - 1):
                    if rmin <= r <= rmax and cmin <= c <= cmax:
                        continue
                    if rmin <= r <= rmax and cmin <= c + 1 <= cmax:
                        continue

                    left = int(arr[r, c])
                    right = int(arr[r, c + 1])
                    if left == 0 or right == 0 or left == right:
                        continue

                    if mode == "panel_on_right":
                        if right in panel_colors:
                            mapping[right] = left
                    else:
                        if left in panel_colors:
                            mapping[left] = right

            if not mapping:
                return None

            out = panel.copy()
            for old_color, new_color in mapping.items():
                out[out == old_color] = new_color
            return out

        # Learn panel selection rank + pair orientation by exact training match.
        learned_mode: str | None = None
        learned_rank: int | None = None

        for ts in arc_problem.training_set():
            train_in = ts.get_input_data().data()
            train_out = ts.get_output_data().data()

            candidates = get_solid_panel_candidates(train_in)
            if not candidates:
                return None

            matched: list[tuple[int, str]] = []
            for rank, box in enumerate(candidates):
                for mode in ("panel_on_right", "panel_on_left"):
                    out = make_output_for_box(train_in, box, mode)
                    if out is None:
                        continue
                    if out.shape == train_out.shape and np.array_equal(out, train_out):
                        matched.append((rank, mode))

            if len(matched) != 1:
                return None

            this_rank, this_mode = matched[0]
            if learned_rank is None:
                learned_rank = this_rank
            elif learned_rank != this_rank:
                return None

            if learned_mode is None:
                learned_mode = this_mode
            elif learned_mode != this_mode:
                return None

        if learned_rank is None or learned_mode is None:
            return None

        # Apply learned rule to test input.
        test_candidates = get_solid_panel_candidates(grid)
        if learned_rank >= len(test_candidates):
            return None

        return make_output_for_box(grid, test_candidates[learned_rank], learned_mode)

    def solve_67c52801_pack_colors_into_support_slots_dynamic(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve 67c52801-like tasks dynamically:
        - bottom row is a solid base color
        - row above contains support posts (base color) and zero-valued slots
        - non-base colors above are packed into these slots as filled rectangles resting on support row.
        """
        rows, cols = grid.shape
        if rows < 2:
            return None

        bottom = grid[rows - 1, :]
        if not np.all(bottom == bottom[0]) or int(bottom[0]) == 0:
            return None
        base_color = int(bottom[0])
        support_row = grid[rows - 2, :]

        # Find zero slots on support row.
        slots: list[tuple[int, int]] = []
        c = 0
        while c < cols:
            if support_row[c] != 0:
                c += 1
                continue
            start = c
            while c < cols and support_row[c] == 0:
                c += 1
            slots.append((start, c - 1))
        if not slots:
            return None

        # Count each non-base, non-zero color above the base row.
        color_counts: dict[int, int] = {}
        for r in range(rows - 1):
            for c in range(cols):
                color = int(grid[r, c])
                if color in (0, base_color):
                    continue
                color_counts[color] = color_counts.get(color, 0) + 1
        if len(color_counts) != len(slots):
            return None

        sorted_slots = sorted(slots, key=lambda s: (s[1] - s[0] + 1, s[0]))
        sorted_colors = sorted(color_counts.items(), key=lambda kv: kv[1])

        out = np.zeros_like(grid)
        out[rows - 1, :] = base_color
        out[rows - 2, support_row != 0] = base_color

        for (start, end), (color, count) in zip(sorted_slots, sorted_colors):
            width = end - start + 1
            if count % width != 0:
                return None
            height = count // width
            top = rows - 2 - height + 1
            if top < 0:
                return None
            out[top : rows - 1, start : end + 1] = color

        return out

    def solve_c48954c1_reflective_block_tiling(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve c48954c1-like tasks dynamically:
        create a 3x3 block tiling from the input square using a 180-rotated base and its
        horizontal/vertical reflections:
            [ A, H(A), A
              V(A), HV(A), V(A)
              A, H(A), A ]
        """
        rows, cols = grid.shape
        if rows != cols or rows == 0:
            return None

        a = np.rot90(grid, 2)
        h = np.fliplr(a)
        v = np.flipud(a)
        hv = np.flipud(h)
        return np.block([[a, h, a], [v, hv, v], [a, h, a]])

    def solve_c1990cce_generate_v_and_mod4_diagonals(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve c1990cce-like tasks:
        from a 1xN row with a single color-2 seed, generate an NxN output with:
          - a top V-shape in color 2
          - a lower diagonal lattice in color 1 using a dynamic mod-4 rule and expanding window.
        """
        rows, cols = grid.shape
        if rows != 1:
            return None

        seeds = np.argwhere(grid == 2)
        if len(seeds) != 1:
            return None
        _, p = map(int, seeds[0])
        n = cols

        out = np.zeros((n, n), dtype=int)

        # Color-2 V shape.
        for r in range(0, p + 1):
            c1 = p - r
            c2 = p + r
            if 0 <= c1 < n:
                out[r, c1] = 2
            if 0 <= c2 < n:
                out[r, c2] = 2

        # Color-1 lattice, dynamic by row and centered expanding window.
        for r in range(3, n):
            left = max(0, p - (r - 2))
            right = min(n - 1, p + (r - 2))
            residue = (r + p - 4) % 4
            for c in range(left, right + 1):
                if c % 4 == residue and out[r, c] == 0:
                    out[r, c] = 1

        return out

    def solve_e98196ab_overlay_halves_across_divider(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve e98196ab-like tasks:
        split on a full-width non-zero divider row, then overlay bottom half onto top half to produce
        a compressed output of half-height.
        """
        rows, cols = grid.shape
        divider_rows = [
            r for r in range(rows)
            if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != 0
        ]
        if len(divider_rows) != 1:
            return None
        d = divider_rows[0]

        top = grid[:d, :]
        bottom = grid[d + 1 :, :]
        if top.shape != bottom.shape or top.shape[0] == 0:
            return None

        out = top.copy()
        mask = (out == 0) & (bottom != 0)
        out[mask] = bottom[mask]
        return out

    def solve_dc433765_move_three_toward_four(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """Solve dc433765-like tasks by moving the single 3 one step toward the single 4."""
        pos3 = np.argwhere(grid == 3)
        pos4 = np.argwhere(grid == 4)
        if len(pos3) != 1 or len(pos4) != 1:
            return None

        r3, c3 = map(int, pos3[0])
        r4, c4 = map(int, pos4[0])
        dr = 0 if r4 == r3 else (1 if r4 > r3 else -1)
        dc = 0 if c4 == c3 else (1 if c4 > c3 else -1)

        nr, nc = r3 + dr, c3 + dc
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            return None

        out = grid.copy()
        out[r3, c3] = 0
        out[nr, nc] = 3
        return out

    def solve_f35d900a_corner_blocks_with_dashed_connectors(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve f35d900a-like tasks:
        four colored corner seeds define a rectangle. Expand each seed into a 3x3 block filled with
        the opposite seed color, keep seed centers unchanged, and connect the four blocks with color-5
        horizontal/vertical connectors (dashed for long gaps).
        """
        pts = np.argwhere(grid != 0)
        if len(pts) != 4:
            return None

        rows = sorted({int(r) for r, _ in pts})
        cols = sorted({int(c) for _, c in pts})
        if len(rows) != 2 or len(cols) != 2:
            return None
        r_top, r_bot = rows
        c_left, c_right = cols

        expected_corners = {(r_top, c_left), (r_top, c_right), (r_bot, c_left), (r_bot, c_right)}
        if {tuple(map(int, p)) for p in pts} != expected_corners:
            return None

        seed_colors = sorted({int(grid[r, c]) for r, c in expected_corners})
        if len(seed_colors) != 2:
            return None

        out = np.zeros_like(grid)

        # Expand each corner seed into a 3x3 block with the opposite color and preserve center.
        for r, c in expected_corners:
            center_color = int(grid[r, c])
            fill_color = seed_colors[0] if center_color == seed_colors[1] else seed_colors[1]
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                        out[rr, cc] = fill_color
            out[r, c] = center_color

        def _paint_horizontal_from_both_sides(row: int):
            mid = (c_left + c_right) / 2.0
            c = c_left + 2
            while c <= c_right - 2:
                if c > mid:
                    break
                if out[row, c] == 0:
                    out[row, c] = 5
                c += 2
            c = c_right - 2
            while c >= c_left + 2:
                if c < mid:
                    break
                if out[row, c] == 0:
                    out[row, c] = 5
                c -= 2

        def _paint_vertical_from_both_sides(col: int):
            mid = (r_top + r_bot) / 2.0
            r = r_top + 2
            while r <= r_bot - 2:
                if r > mid:
                    break
                if out[r, col] == 0:
                    out[r, col] = 5
                r += 2
            r = r_bot - 2
            while r >= r_top + 2:
                if r < mid:
                    break
                if out[r, col] == 0:
                    out[r, col] = 5
                r -= 2

        # Connectors between corresponding corner blocks.
        _paint_horizontal_from_both_sides(r_top)
        _paint_horizontal_from_both_sides(r_bot)
        _paint_vertical_from_both_sides(c_left)
        _paint_vertical_from_both_sides(c_right)

        return out

    def solve_f8a8fe49_reflect_fives_across_twos(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve f8a8fe49-like tasks:
        keep the color-2 scaffold fixed, and reflect each color-5 pixel outward across one of two
        opposite scaffold walls (horizontal or vertical), chosen by side of the scaffold midpoint.
        """
        two_pts = np.argwhere(grid == 2)
        five_pts = np.argwhere(grid == 5)
        if len(two_pts) == 0 or len(five_pts) == 0:
            return None

        rmin, cmin = two_pts.min(axis=0)
        rmax, cmax = two_pts.max(axis=0)
        out = np.zeros_like(grid)
        out[grid == 2] = 2

        # Determine scaffold orientation.
        top_full = np.all(grid[rmin, cmin : cmax + 1] == 2)
        bottom_full = np.all(grid[rmax, cmin : cmax + 1] == 2)
        horizontal_mode = bool(top_full and bottom_full)

        if horizontal_mode:
            mid = (rmin + rmax) / 2.0
            for r, c in five_pts:
                rr = int(2 * rmin - r) if r <= mid else int(2 * rmax - r)
                if 0 <= rr < grid.shape[0]:
                    out[rr, c] = 5
        else:
            left_full = np.all(grid[rmin : rmax + 1, cmin] == 2)
            right_full = np.all(grid[rmin : rmax + 1, cmax] == 2)
            if not (left_full and right_full):
                return None
            mid = (cmin + cmax) / 2.0
            for r, c in five_pts:
                cc = int(2 * cmin - c) if c <= mid else int(2 * cmax - c)
                if 0 <= cc < grid.shape[1]:
                    out[r, cc] = 5

        return out

    def solve_cf98881b_overlay_three_panels_left_priority(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve cf98881b-like tasks:
        input is three equal-width panels separated by two uniform divider columns.
        Output is a left-priority overlay: panel1 cell if non-zero, else panel2, else panel3.
        """
        rows, cols = grid.shape
        if cols < 5:
            return None

        divider_pairs = []
        for d1 in range(1, cols - 2):
            for d2 in range(d1 + 1, cols - 1):
                c1 = grid[:, d1]
                c2 = grid[:, d2]
                if not (np.all(c1 == c1[0]) and np.all(c2 == c2[0])):
                    continue
                if int(c1[0]) == 0 or int(c1[0]) != int(c2[0]):
                    continue
                left_w = d1
                mid_w = d2 - d1 - 1
                right_w = cols - d2 - 1
                if not (left_w == mid_w == right_w):
                    continue
                divider_pairs.append((d1, d2))

        if len(divider_pairs) != 1:
            return None
        d1, d2 = divider_pairs[0]

        left = grid[:, :d1]
        mid = grid[:, d1 + 1 : d2]
        right = grid[:, d2 + 1 :]
        if not (left.shape == mid.shape == right.shape):
            return None

        out = np.where(left != 0, left, np.where(mid != 0, mid, right))
        return out.astype(int)

    def solve_74dd1130_transpose_grid(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """Solve 74dd1130-like tasks by transposing the grid."""
        rows, cols = grid.shape
        if rows != cols:
            return None
        return grid.T.copy()

    def solve_f2829549_nor_left_right_halves(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve f2829549-like tasks:
        split by a single full-height separator column. Compare left and right halves cell-wise
        (same row/column index within each half) and mark output where both halves are background.
        """
        rows, cols = grid.shape
        if cols < 3:
            return None

        separator_candidates = []
        for c in range(1, cols - 1):
            if not (np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != 0):
                continue
            if c != cols - c - 1:
                continue
            separator_candidates.append(c)
        if len(separator_candidates) != 1:
            return None

        sep = separator_candidates[0]
        left = grid[:, :sep]
        right = grid[:, sep + 1 :]
        if left.shape != right.shape:
            return None

        out_color = self._learn_nonzero_output_color(arc_problem)
        if out_color is None:
            return None

        out = np.zeros(left.shape, dtype=int)
        out[(left == 0) & (right == 0)] = out_color
        return out

    def solve_f25ffba3_reflect_lower_half_to_top(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve f25ffba3-like tasks:
        find the first non-zero row (start of the motif). The prefix above it is blank, and the
        lower block from that row to the bottom is mirrored upward into the blank prefix.
        The original lower block remains unchanged.
        """
        rows, _ = grid.shape
        nonzero_rows = np.where(np.any(grid != 0, axis=1))[0]
        if len(nonzero_rows) == 0:
            return None

        start = int(nonzero_rows[0])
        if start == 0:
            return None

        if not np.all(grid[:start, :] == 0):
            return None

        lower = grid[start:, :]
        if lower.shape[0] != start:
            return None

        out = grid.copy()
        out[:start, :] = lower[::-1, :]
        return out

    def solve_d687bc17_route_edge_colors_to_inner_border(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve d687bc17-like tasks:
        the outer border defines four edge colors (top/bottom/left/right). For every interior
        occurrence of one of those colors, keep that color only on the corresponding inner border
        lane (row=1, row=H-2, col=1, col=W-2) at the same column/row; remove all other interior noise.
        """
        rows, cols = grid.shape
        if rows < 3 or cols < 3:
            return None

        top_color = int(grid[0, 1])
        bottom_color = int(grid[rows - 1, 1])
        left_color = int(grid[1, 0])
        right_color = int(grid[1, cols - 1])

        # Require a consistent framed border with zero-valued corners.
        if not (
            int(grid[0, 0]) == 0
            and int(grid[0, cols - 1]) == 0
            and int(grid[rows - 1, 0]) == 0
            and int(grid[rows - 1, cols - 1]) == 0
        ):
            return None
        if not (
            np.all(grid[0, 1 : cols - 1] == top_color)
            and np.all(grid[rows - 1, 1 : cols - 1] == bottom_color)
            and np.all(grid[1 : rows - 1, 0] == left_color)
            and np.all(grid[1 : rows - 1, cols - 1] == right_color)
        ):
            return None

        out = np.zeros_like(grid)
        out[0, :] = grid[0, :]
        out[rows - 1, :] = grid[rows - 1, :]
        out[:, 0] = grid[:, 0]
        out[:, cols - 1] = grid[:, cols - 1]

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                color = int(grid[r, c])
                if color == top_color:
                    out[1, c] = top_color
                elif color == bottom_color:
                    out[rows - 2, c] = bottom_color
                elif color == left_color:
                    out[r, 1] = left_color
                elif color == right_color:
                    out[r, cols - 2] = right_color

        return out




    def solve_f76d97a5_project_fives_to_other_color(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve f76d97a5-like tasks:
        keep only cells equal to color 5, but repaint them using the other input color;
        all non-5 cells become 0.
        """
        colors = [int(c) for c in np.unique(grid)]
        if 5 not in colors:
            return None
        others = [c for c in colors if c not in (0, 5)]
        if len(others) != 1:
            return None

        out_color = others[0]
        out = np.zeros_like(grid)
        out[grid == 5] = out_color
        return out

    def solve_ce4f8723_or_top_bottom_masks(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve ce4f8723-like tasks:
        split into top 4x4 and bottom 4x4 masks (separated by a solid row), then output
        the cell-wise OR mask painted with the learned output color.
        """
        rows, cols = grid.shape
        if rows != 9 or cols != 4:
            return None

        # separator row should be uniform non-zero
        sep_row = grid[4, :]
        if not (np.all(sep_row == sep_row[0]) and int(sep_row[0]) != 0):
            return None

        top = grid[:4, :]
        bottom = grid[5:, :]
        if top.shape != (4, 4) or bottom.shape != (4, 4):
            return None

        out_color = self._learn_nonzero_output_color(arc_problem)
        if out_color is None:
            return None

        mask = (top != 0) | (bottom != 0)
        out = np.zeros((4, 4), dtype=int)
        out[mask] = out_color
        return out

    def solve_ce22a75a_expand_points_to_3x3(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve ce22a75a-like tasks:
        each non-zero seed point becomes a 3x3 filled block centered on that seed,
        using the learned non-zero output color.
        """
        seeds = np.argwhere(grid != 0)
        if len(seeds) == 0:
            return None

        out_color = self._learn_nonzero_output_color(arc_problem)
        if out_color is None:
            return None

        rows, cols = grid.shape
        out = np.zeros_like(grid)

        for r0, c0 in seeds:
            r0, c0 = int(r0), int(c0)
            for r in range(r0 - 1, r0 + 2):
                for c in range(c0 - 1, c0 + 2):
                    if 0 <= r < rows and 0 <= c < cols:
                        out[r, c] = out_color

        return out

    def solve_b94a9452_swap_inner_outer(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve b94a9452-like tasks:
        detect a rectangular non-zero block with an inner motif of a second color,
        crop to the block, and swap colors (outer<->inner) within that crop.
        """
        nz = np.argwhere(grid != 0)
        if len(nz) == 0:
            return None

        rmin, cmin = nz.min(axis=0)
        rmax, cmax = nz.max(axis=0)
        block = grid[rmin : rmax + 1, cmin : cmax + 1]

        colors = [int(c) for c in np.unique(block) if int(c) != 0]
        if len(colors) != 2:
            return None

        c1, c2 = colors
        n1 = int(np.sum(block == c1))
        n2 = int(np.sum(block == c2))

        # Outer color is the majority in the cropped block.
        outer = c1 if n1 > n2 else c2
        inner = c2 if outer == c1 else c1

        out = np.full(block.shape, inner, dtype=int)
        out[block == inner] = outer
        return out

    def solve_623ea044_draw_diagonals_through_point(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve 623ea044-like tasks:
        from a single colored seed pixel, draw both diagonals (an X) across the grid.
        """
        pts = np.argwhere(grid != 0)
        if len(pts) != 1:
            return None

        r0, c0 = map(int, pts[0])
        color = int(grid[r0, c0])
        rows, cols = grid.shape

        out = np.zeros_like(grid)

        # main diagonal: (r-k, c-k) and (r+k, c+k)
        for k in range(-max(rows, cols), max(rows, cols) + 1):
            r = r0 + k
            c = c0 + k
            if 0 <= r < rows and 0 <= c < cols:
                out[r, c] = color

        # anti-diagonal: (r-k, c+k) and (r+k, c-k)
        for k in range(-max(rows, cols), max(rows, cols) + 1):
            r = r0 + k
            c = c0 - k
            if 0 <= r < rows and 0 <= c < cols:
                out[r, c] = color

        return out

    def solve_28e73c20_spiral_maze(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve 28e73c20-like tasks (blank input -> deterministic spiral maze pattern).
        Draw a single-pixel-width spiral path with one-cell corridors using color 3.
        """
        if np.any(grid != 0):
            return None
        rows, cols = grid.shape
        if rows != cols:
            return None

        n = rows
        out = np.zeros((n, n), dtype=int)
        path_color = self._learn_nonzero_output_color(arc_problem)
        if path_color is None:
            path_color = 3

        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        d = 0
        r = c = 0
        out[r, c] = path_color

        def can_step(nr: int, nc: int, pr: int, pc: int) -> bool:
            if not (0 <= nr < n and 0 <= nc < n):
                return False
            if int(out[nr, nc]) != 0:
                return False
            # Keep one-cell corridor from existing path (except the previous cell).
            for dr, dc in dirs:
                ar, ac = nr + dr, nc + dc
                if 0 <= ar < n and 0 <= ac < n and int(out[ar, ac]) == path_color and not (ar == pr and ac == pc):
                    return False
            return True

        while True:
            moved = False
            for nd in (d, (d + 1) % 4):
                dr, dc = dirs[nd]
                nr, nc = r + dr, c + dc
                if can_step(nr, nc, r, c):
                    d = nd
                    r, c = nr, nc
                    out[r, c] = path_color
                    moved = True
                    break
            if not moved:
                break

        # Even-sized grids in this family include one extra center-left link cell.
        if n % 2 == 0:
            out[n // 2, n // 2 - 1] = path_color

        return out

    def solve_bbb1b8b6_overlay_if_disjoint(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve bbb1b8b6-like tasks:
        split around the middle separator column, then either:
        - return the left block unchanged if left/right non-zero pixels overlap, or
        - overlay right colors onto zero cells of the left block when they are disjoint.
        """
        rows, cols = grid.shape
        sep_cols = [c for c in range(cols) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != 0]
        mid_candidates = [c for c in sep_cols if c == (cols - 1 - c)]
        if len(mid_candidates) != 1:
            return None
        sep = mid_candidates[0]

        left = grid[:, :sep]
        right = grid[:, sep + 1 :]
        if left.shape != right.shape:
            return None

        overlap = np.any((left != 0) & (right != 0))
        if overlap:
            return left.copy()

        out = left.copy()
        mask = (out == 0) & (right != 0)
        out[mask] = right[mask]
        return out

    def solve_992798f6_dominant_axis_path(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve 992798f6-like tasks:
        two singleton endpoint colors are connected with a third color path.
        Build path from a cell adjacent to the start endpoint toward a cell adjacent to
        the end endpoint, moving first along the dominant axis and then diagonally.
        """
        nonzero = [int(c) for c in np.unique(grid) if int(c) != 0]
        if len(nonzero) != 2:
            return None

        pos_by_color = {}
        for c in nonzero:
            pts = np.argwhere(grid == c)
            if len(pts) != 1:
                return None
            pos_by_color[c] = tuple(map(int, pts[0]))

        # In this family, start endpoint is color 2, end endpoint is color 1.
        if 2 in pos_by_color and 1 in pos_by_color:
            start = pos_by_color[2]
            end = pos_by_color[1]
        else:
            # fallback deterministic ordering
            c_sorted = sorted(nonzero)
            start = pos_by_color[c_sorted[-1]]
            end = pos_by_color[c_sorted[0]]

        path_color = self._learn_added_output_color(arc_problem)
        if path_color is None:
            return None

        r0, c0 = start
        r1, c1 = end
        dr = 1 if r1 > r0 else -1 if r1 < r0 else 0
        dc = 1 if c1 > c0 else -1 if c1 < c0 else 0

        cur_r, cur_c = r0 + dr, c0 + dc
        tgt_r, tgt_c = r1 - dr, c1 - dc

        out = grid.copy()

        rows, cols = grid.shape
        def in_bounds(r: int, c: int) -> bool:
            return 0 <= r < rows and 0 <= c < cols

        while (cur_r, cur_c) != (tgt_r, tgt_c):
            if in_bounds(cur_r, cur_c) and int(out[cur_r, cur_c]) == 0:
                out[cur_r, cur_c] = path_color

            rem_r = tgt_r - cur_r
            rem_c = tgt_c - cur_c
            abs_r = abs(rem_r)
            abs_c = abs(rem_c)

            step_r = 0 if rem_r == 0 else (1 if rem_r > 0 else -1)
            step_c = 0 if rem_c == 0 else (1 if rem_c > 0 else -1)

            if abs_r > abs_c:
                cur_r += step_r
            elif abs_c > abs_r:
                cur_c += step_c
            else:
                cur_r += step_r
                cur_c += step_c

        if in_bounds(tgt_r, tgt_c) and int(out[tgt_r, tgt_c]) == 0:
            out[tgt_r, tgt_c] = path_color

        return out

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
        split by full separator rows; for adjacent row-segments where one segment's color set
        is a strict subset of the other's, copy a vertical flip of the richer segment into the
        poorer one (non-separator columns only).
        """
        rows, cols = grid.shape

        full_row_colors = [int(grid[r, 0]) for r in range(rows) if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != 0]
        full_col_colors = [int(grid[0, c]) for c in range(cols) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != 0]
        sep_candidates = set(full_row_colors) & set(full_col_colors)
        if len(sep_candidates) != 1:
            return None
        sep = next(iter(sep_candidates))

        divider_rows = [r for r in range(rows) if np.all(grid[r, :] == sep)]
        segments: list[tuple[int, int]] = []
        prev = -1
        for dr in divider_rows + [rows]:
            r0, r1 = prev + 1, dr
            if r0 < r1:
                segments.append((r0, r1))
            prev = dr
        if len(segments) < 2:
            return None

        sep_cols = {c for c in range(cols) if np.all(grid[:, c] == sep)}
        out = grid.copy()

        def color_set(r0: int, r1: int) -> set[int]:
            vals = {int(v) for v in np.unique(grid[r0:r1, :])}
            vals.discard(0)
            vals.discard(sep)
            return vals

        for i in range(len(segments) - 1):
            a0, a1 = segments[i]
            b0, b1 = segments[i + 1]
            if (a1 - a0) != (b1 - b0):
                continue

            set_a = color_set(a0, a1)
            set_b = color_set(b0, b1)
            if not set_a and not set_b:
                continue

            src = None
            tgt = None
            cnt_a = int(np.sum((grid[a0:a1, :] != 0) & (grid[a0:a1, :] != sep)))
            cnt_b = int(np.sum((grid[b0:b1, :] != 0) & (grid[b0:b1, :] != sep)))

            if set_a == set_b and set_a:
                if cnt_a > cnt_b:
                    src, tgt = (a0, a1), (b0, b1)
                elif cnt_b > cnt_a:
                    src, tgt = (b0, b1), (a0, a1)
                else:
                    continue
            elif set_a < set_b and set_a:  # A strict subset of B, A not empty
                src, tgt = (b0, b1), (a0, a1)
            elif set_b < set_a and set_b:  # B strict subset of A, B not empty
                src, tgt = (a0, a1), (b0, b1)
            else:
                continue

            sr0, sr1 = src
            tr0, tr1 = tgt
            src_flip = np.flipud(grid[sr0:sr1, :])

            for c in range(cols):
                if c in sep_cols:
                    continue
                out[tr0:tr1, c] = src_flip[:, c]

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
        def detect_centers_by_color(arr: np.ndarray) -> dict[int, list[tuple[int, int]]]:
            rr, cc = arr.shape
            grouped: dict[int, list[tuple[int, int]]] = {}
            for r in range(1, rr - 1):
                for c in range(1, cc - 1):
                    if int(arr[r, c]) != 0:
                        continue
                    up = int(arr[r - 1, c])
                    down = int(arr[r + 1, c])
                    left = int(arr[r, c - 1])
                    right = int(arr[r, c + 1])
                    if up != 0 and up == down == left == right:
                        grouped.setdefault(up, []).append((r, c))
            return grouped

        def apply_connectors(arr: np.ndarray, centers: list[tuple[int, int]], connector: int) -> np.ndarray:
            out = arr.copy()

            by_row: dict[int, list[int]] = {}
            for r, c in centers:
                by_row.setdefault(r, []).append(c)

            for same_row, col_positions in by_row.items():
                col_positions.sort()
                for i in range(len(col_positions) - 1):
                    left_center_col = col_positions[i]
                    right_center_col = col_positions[i + 1]
                    start = left_center_col + 2
                    stop = right_center_col - 1
                    if start >= stop:
                        continue

                    can_connect = True
                    for c in range(start, stop):
                        if int(arr[same_row, c]) != 0:
                            can_connect = False
                            break
                    if can_connect:
                        for c in range(start, stop):
                            out[same_row, c] = connector

            by_col: dict[int, list[int]] = {}
            for r, c in centers:
                by_col.setdefault(c, []).append(r)

            for same_col, row_positions in by_col.items():
                row_positions.sort()
                for i in range(len(row_positions) - 1):
                    top_center_row = row_positions[i]
                    bottom_center_row = row_positions[i + 1]
                    start = top_center_row + 2
                    stop = bottom_center_row - 1
                    if start >= stop:
                        continue

                    can_connect = True
                    for r in range(start, stop):
                        if int(arr[r, same_col]) != 0:
                            can_connect = False
                            break
                    if can_connect:
                        for r in range(start, stop):
                            out[r, same_col] = connector

            return out

        # Learn connector color.
        line_color = self._learn_added_output_color(arc_problem)

        # Learn motif color by exact training simulation (robust to distractor motifs/colors).
        learned_motif_color: int | None = None
        if line_color is not None:
            for ts in arc_problem.training_set():
                train_in = ts.get_input_data().data()
                train_out = ts.get_output_data().data()
                centers_by_color = detect_centers_by_color(train_in)
                if not centers_by_color:
                    return None

                matched_for_this_example: list[int] = []
                for motif_color, centers in centers_by_color.items():
                    candidate = apply_connectors(train_in, centers, line_color)
                    if np.array_equal(candidate, train_out):
                        matched_for_this_example.append(motif_color)

                if len(matched_for_this_example) != 1:
                    return None

                this_color = matched_for_this_example[0]
                if learned_motif_color is None:
                    learned_motif_color = this_color
                elif learned_motif_color != this_color:
                    return None

        # Fallback: infer connector color from changed cells once motif color is known.
        if learned_motif_color is None:
            for ts in arc_problem.training_set():
                train_in = ts.get_input_data().data()
                train_out = ts.get_output_data().data()
                centers_by_color = detect_centers_by_color(train_in)
                if not centers_by_color:
                    return None

                best_color = None
                for motif_color, centers in centers_by_color.items():
                    changed = np.argwhere((train_in == 0) & (train_out != 0))
                    inferred = {int(train_out[r, c]) for r, c in changed if int(train_out[r, c]) != motif_color}
                    if len(inferred) != 1:
                        continue
                    candidate = apply_connectors(train_in, centers, next(iter(inferred)))
                    if np.array_equal(candidate, train_out):
                        if best_color is not None:
                            return None
                        best_color = motif_color
                        line_color = next(iter(inferred))

                if best_color is None or line_color is None:
                    return None
                if learned_motif_color is None:
                    learned_motif_color = best_color
                elif learned_motif_color != best_color:
                    return None

        centers_by_color = detect_centers_by_color(grid)
        if learned_motif_color is None or line_color is None or learned_motif_color not in centers_by_color:
            return None
        return apply_connectors(grid, centers_by_color[learned_motif_color], line_color)

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
