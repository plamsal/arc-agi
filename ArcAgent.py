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
