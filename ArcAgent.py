import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init. Be aware that it gets called only once
        and then the solve method will get called several times.
        """
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        Write the code in this method
        to solve the incoming ArcProblem.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer.

        Also, you shouldn't add more than 3 predictions to the list as
        that is considered an ERROR and the test will be automatically
        marked as incorrect.
        """

        predictions: list[np.ndarray] = list()

        test_input = arc_problem.test_set().get_input_data().data()

        result = self.solve_connecting_crosses(test_input, arc_problem)
        if result is not None:
            predictions.append(result)

        if not predictions:
            predictions.append(test_input.copy())

        return predictions

    def solve_connecting_crosses(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Generalized solver for problems where same-colored cross/plus shapes are connected
        by a new connector color. Works with any background color, any arm length, and
        both hollow crosses (background center) and solid crosses (colored center).
        """

        def learn_colors():
            bg_color = None
            line_color = None
            for ts in arc_problem.training_set():
                train_in  = ts.get_input_data().data()
                train_out = ts.get_output_data().data()

                vals, counts = np.unique(train_in, return_counts=True)
                bg = int(vals[np.argmax(counts)])
                if bg_color is None:
                    bg_color = bg
                elif bg_color != bg:
                    return None, None

                in_colors  = {int(c) for c in np.unique(train_in)  if int(c) != bg_color}
                out_colors = {int(c) for c in np.unique(train_out) if int(c) != bg_color}
                added = out_colors - in_colors
                if len(added) != 1:
                    return None, None
                lc = next(iter(added))
                if line_color is None:
                    line_color = lc
                elif line_color != lc:
                    return None, None
            return bg_color, line_color

        bg_color, learned_line_color = learn_colors()
        if bg_color is None or learned_line_color is None:
            return None

        def arm_extent(my_grid, r, c, dr, dc):
            """Returns (color, length) of the arm starting one step from (r,c) in direction (dr,dc)."""
            my_rows, my_cols = my_grid.shape
            nr, nc = r + dr, c + dc
            if not (0 <= nr < my_rows and 0 <= nc < my_cols):
                return None, 0
            arm_color = int(my_grid[nr, nc])
            if arm_color == bg_color:
                return None, 0
            length = 0
            while 0 <= nr < my_rows and 0 <= nc < my_cols and int(my_grid[nr, nc]) == arm_color:
                length += 1
                nr += dr
                nc += dc
            return arm_color, length

        def find_all_cross_centers(my_grid):
            """
            Finds cross centers in two forms:
            - Hollow: background cell with same-color arms extending in all 4 directions.
            - Solid: colored cell with same-color arms (same color as center) in all 4 directions.
            Returns dict mapping cross_color -> list of (row, col, up_len, down_len, left_len, right_len).
            """
            my_rows, my_cols = my_grid.shape
            centers = {}
            for r in range(my_rows):
                for c in range(my_cols):
                    cell = int(my_grid[r, c])
                    up_color,    ul = arm_extent(my_grid, r, c, -1,  0)
                    down_color,  dl = arm_extent(my_grid, r, c,  1,  0)
                    left_color,  ll = arm_extent(my_grid, r, c,  0, -1)
                    right_color, rl = arm_extent(my_grid, r, c,  0,  1)

                    if not (up_color and down_color and left_color and right_color):
                        continue

                    if cell == bg_color:
                        # Hollow cross: background center, same-color arms in all 4 directions
                        if up_color == down_color == left_color == right_color:
                            cross_color = up_color
                            centers.setdefault(cross_color, []).append((r, c, ul, dl, ll, rl))
                    else:
                        # Solid cross: colored center, arms must match center color in all 4 directions
                        if up_color == down_color == left_color == right_color == cell:
                            centers.setdefault(cell, []).append((r, c, ul, dl, ll, rl))

            return centers

        def connect_all_crosses(my_grid, centers, line_color):
            result = my_grid.copy()
            for cross_color, center_list in centers.items():

                # Horizontal connections (same row)
                by_row = {}
                for r, c, ul, dl, ll, rl in center_list:
                    by_row.setdefault(r, []).append((c, ll, rl))

                for row, items in by_row.items():
                    items.sort()
                    for i in range(len(items) - 1):
                        lc, _, l_rl = items[i]
                        rc, r_ll, _ = items[i + 1]
                        fill_start = lc + l_rl + 1
                        fill_end   = rc - r_ll
                        if fill_start >= fill_end:
                            continue
                        if all(int(my_grid[row, cc]) == bg_color for cc in range(fill_start, fill_end)):
                            for cc in range(fill_start, fill_end):
                                result[row, cc] = line_color

                # Vertical connections (same col)
                by_col = {}
                for r, c, ul, dl, ll, rl in center_list:
                    by_col.setdefault(c, []).append((r, ul, dl))

                for col, items in by_col.items():
                    items.sort()
                    for i in range(len(items) - 1):
                        tr, _, t_dl = items[i]
                        br, b_ul, _ = items[i + 1]
                        fill_start = tr + t_dl + 1
                        fill_end   = br - b_ul
                        if fill_start >= fill_end:
                            continue
                        if all(int(my_grid[rr, col]) == bg_color for rr in range(fill_start, fill_end)):
                            for rr in range(fill_start, fill_end):
                                result[rr, col] = line_color

            return result

        # Validate on all training pairs
        for my_ts in arc_problem.training_set():
            train_in  = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()
            my_centers = find_all_cross_centers(train_in)
            if not my_centers:
                return None
            if not np.array_equal(connect_all_crosses(train_in, my_centers, learned_line_color), train_out):
                return None

        # Apply to test grid
        my_centers = find_all_cross_centers(grid)
        if not my_centers:
            return None
        output = connect_all_crosses(grid, my_centers, learned_line_color)
        if np.array_equal(output, grid):
            return None
        return output
