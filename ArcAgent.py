from collections import deque

import numpy as np

from ArcProblem import ArcProblem


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        test_input = arc_problem.test_set().get_input_data().data()

        prediction = self._fill_closed_regions_with_hint_majority(test_input)
        return [prediction]

    def _fill_closed_regions_with_hint_majority(self, grid: np.ndarray) -> np.ndarray:
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
