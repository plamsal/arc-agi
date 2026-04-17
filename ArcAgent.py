
from collections import deque

# 0 black, 1 blue, 2 red, 3 green, 4 yellow, 7 orange, 8 = Azure/Light Blue, 9 = Maroon
# Generalized agent: learns colors, separators, and logic from training data
# No more hardcoded colors — the hidden tests use different colors for the same patterns

import time  


import numpy as np
from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet








class ArcAgent:
    def __init__(self):
        self._total_time = 0.0
        self._problem_count = 0
        # """
        # Initialize the agent. This is called only once.
        # """
        # pass




    #---------------------------------------------
    # My MAIN ENTRY POINT
    #---------------------------------------------


    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        _start = time.perf_counter()
        predictions: list[np.ndarray] = list()
        test_input = arc_problem.test_set().get_input_data().data()

        strategies = [
            self.match_left_right_and_color,
            self.match_top_and_bottom_color,
            self.hollow_out_center,
            self.crop_colored_region,
            self.remap_colors,
            self.rotate_or_flip,
            self.fill_any_enclosed_loop,
            self.expand_staircase,
            self.recolor_center_with_corner_color,
            self.diagonal_line_move,
            self.sort_colors_left_to_right_by_count,
            self.connect_matching_blocks,
            self.lone_color_extension,
            self.mirror_my_input,
            self.split_and_recolor_using_xor,
            self.fill_closed_regions_with_hint_majority,
            self.solve_comparing_xor_halves,
            self.solve_connecting_crosses,
            self.solve_histogram_from_frequent_patch,
            self.solve_inner_fill_with_reflection_in_frames,
            self.solve_with_or_halves,
            self.overlay_on_empty,
            
            self.generate_v_with_red_and_blue_diagonals,
            self.output_from_multiple_flipped_blocks,
            self.fit_colored_blocks_into_empty_slots,
            self.extract_the_inside_box_and_recolor_it,
            self.diagonal_and_straight_from_initial_to_final,
            self.count_markers_inside_one_box
         

            
            
        ]

        for strategy in strategies:
            try:
                if not self.validate_on_training(strategy, arc_problem):
                    continue

                strategy_prediction = strategy(test_input, arc_problem)

                if strategy_prediction is not None and strategy_prediction.size > 0:
                    if not any(np.array_equal(strategy_prediction, p) for p in predictions):
                        predictions.append(strategy_prediction)

                if len(predictions) >= 3:
                    break
            except Exception as e:
                continue

        if not predictions:
            predictions.append(test_input.copy())

        _elapsed = time.perf_counter() - _start
        self._total_time += _elapsed
        self._problem_count += 1

        print(f"[{arc_problem.problem_name()}]  {_elapsed*1000:.3f} ms  "
            f"(total so far: {self._total_time*1000:.2f} ms  |  problems: {self._problem_count})")

        return predictions[:3]


    #---------------------------------------------
    # HELPER: Validate a strategy against training
    # If my strategy can't reproduce training outputs, don't trust it on the test
    #---------------------------------------------




    def validate_on_training(self, strategy, arc_problem):
        """
        Run my strategy on every training input.
        Return True only if it reproduces ALL training outputs exactly.
        This prevents wrong strategies from polluting my predictions list.
        """
        for ts in arc_problem.training_set():
            train_in = ts.get_input_data().data()
            train_out = ts.get_output_data().data()




            try:
                result = strategy(train_in, arc_problem)
                if result is None or not np.array_equal(result, train_out):
                    return False
            except:
                return False
        return True




    #---------------------------------------------
    # STRATEGY 1: Vertical Split Match (generalized from 0520fde7)
    # Milestone A hardcoded: separator=5(gray), match=1(blue), output=2(red)
    # New version: learns ALL of that from training examples
    #---------------------------------------------




    def match_left_right_and_color(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:




        # learn overlapping logic from using my training sets from arc_problem list so i can apply to test grid
        all_my_training_sets = arc_problem.training_set()
        training_output_color = None




        for my_ts in all_my_training_sets:
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()




            # Find the separator column in this training input from Milestone A
            # I excluded first and last columns because it caused confuseion in the early test
            # unique value == 1 and not 0 is my separator
            training_sep_col = None
            for col in range(1, train_in.shape[1] - 1):
                col_vals = train_in[:, col]
                unique_vals = np.unique(col_vals)
                if len(unique_vals) == 1 and unique_vals[0] != 0:
                    training_sep_col = col
                    break




            # training set needs separate cols to be considered
            if training_sep_col is None:
                return None  
            # Learn the output color: find any non-zero value from flattened  training output
            non_zero_in_output = train_out[train_out != 0]
            if len(non_zero_in_output) > 0:
                training_output_color = int(non_zero_in_output[0])
                break  




        if training_output_color is None:
            return None




       
        # training_output_color is my assumed color for my output
        # -- Test ---
        # separator should have one color and not 0
        separator_col = None




        for col in range(1, grid.shape[1] - 1):
            col_vals = grid[:, col]
            my_unique_vals = np.unique(col_vals)
            if len(my_unique_vals) == 1 and my_unique_vals[0] != 0:
                separator_col = col
                break




        if separator_col is None:
            return None




        # left and right separator and should be eq
        left_half = grid[:, :separator_col]
        right_half = grid[:, separator_col + 1:]
        if left_half.shape != right_half.shape:
            return None




        # Create my output: where both sides are non-zero → output_color, else 0
        output = np.zeros_like(left_half)




        for row in range(left_half.shape[0]):
            for col in range(left_half.shape[1]):
                left_val = left_half[row, col]
                right_val = right_half[row, col]




                # If both sides have a non-zero pixel at this position -->  then reference my training_output_color
                if left_val != 0 and right_val != 0:
                    output[row, col] = training_output_color
                else:
                    output[row, col] = 0




        return output




    #---------------------------------------------
    # STRATEGY 2: Horizontal Split (generalized — merges overlay + both_black)
    # Milestone A hardcoded: separator=4(yellow), output=3(green)
    # New version: detects separator dynamically, learns OR vs AND logic
    #---------------------------------------------




    def match_top_and_bottom_color(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:




        # -------------- Training ----------------------


        # loop through my training pairs and find the output color
        all_my_training_sets = arc_problem.training_set()
        training_output_color = None


        # I also need to carry OR/AND logic out of the training loop into test
        or_logic_hypothesis = False
        and_logic_hypothesis = False


        for my_ts in all_my_training_sets:
            train_in = my_ts.get_input_data().data() 
            train_out = my_ts.get_output_data().data()




            # Find the separator row in training input
            # I excluded first and last rows because it caused confusion — same reason as vertical
            # unique value == 1 and not 0 is my separator
            training_sep_row  = None


            for row in range(1, train_in.shape[0] - 1):
                row_vals = train_in[row, :]
                unique_vals = np.unique(row_vals)
                if len(unique_vals) == 1 and unique_vals[0] != 0:
                    training_sep_row = row
                    break
            if training_sep_row is None:
                return None
               
            non_zero_in_output = train_out[train_out != 0]
            if len(non_zero_in_output) == 0:
                return None
            training_output_color = int(non_zero_in_output[0])


            top_half_train = train_in[:training_sep_row, :]
            bottom_half_train = train_in[training_sep_row + 1:, :]


            rows = min(top_half_train.shape[0], bottom_half_train.shape[0])
            cols = min(top_half_train.shape[1], bottom_half_train.shape[1])


            or_logic_hypothesis = True
            and_logic_hypothesis = True


            for row in range(rows):
                for col in range(cols):
                    expected = train_out[row, col]
                    either_nonzero = (top_half_train[row, col] != 0 or bottom_half_train[row, col] != 0)
                    both_zero = (top_half_train[row, col] == 0 and bottom_half_train[row, col] == 0)


                    # Check OR logic: does every position produce training_output_color
                    if either_nonzero:
                        if expected != training_output_color:
                            or_logic_hypothesis = False
                    else:
                        if expected != 0:
                            or_logic_hypothesis = False


                    # Check AND logic
                    if both_zero:
                        if expected != training_output_color:
                            and_logic_hypothesis = False
                    else:
                        if expected != 0:
                            and_logic_hypothesis = False
            # if neither hypothesis matches the training, this strategy doesn't apply
            if not or_logic_hypothesis and not and_logic_hypothesis:
                return None


            break  # learned what I need from first valid training set
       
        # training_output_color is my assumed color for my output
        if training_output_color is None:
            return None


        # -------------- Test ----------------------




        # Find the horizontal separator in my test input (same logic as training)
        separator_row   = None
   
        for row in range(1, grid.shape[0] - 1):
            row_vals = grid[row, :]
            my_unique_vals = np.unique(row_vals)
            if len(my_unique_vals) == 1 and my_unique_vals[0] != 0:
                separator_row = row
                break
       
        if separator_row   is None:
            return None






        # Split test input into top and bottom halves
        top_half = grid[:separator_row, :]
        bottom_half = grid[separator_row + 1:, :]
        if top_half.shape != bottom_half.shape:
            return None




        r_count = top_half.shape[0]
        c_count = top_half.shape[1]
        output = np.zeros((r_count, c_count), dtype=int)




        # Apply whichever hypothesis survived training
        for row in range(r_count):
            for col in range(c_count):
                top_val = top_half[row, col]
                bottom_val = bottom_half[row, col]


                # If both sides have a non-zero pixel at this position -> reference my training_output_color
                if or_logic_hypothesis:
                    if top_val != 0 or bottom_val != 0:
                        output[row, col] = training_output_color
                    else:
                        output[row, col] = 0
                elif and_logic_hypothesis:
                    # AND: both sides are black -> training_output_color
                    if top_val == 0 and bottom_val == 0:
                        output[row, col] = training_output_color
                    else:
                        output[row, col] = 0


        return output
   
   
    #---------------------------------------------
    # STRATEGY 3: Hollow Shapes
    # Turn interior pixels black, keep only the border
    # Added: arc_problem param for consistent interface + training validation
    #---------------------------------------------




    def hollow_out_center(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:
        
        # If grid is all black, this strategy doesn't apply
        if np.all(grid == 0):
            return None




        rows, cols = grid.shape
        output = grid.copy()




        # Iterate over every pixel and look for color
        for row in range(rows):
            for col in range(cols):
                color = grid[row, col]
                if color == 0:
                    continue




                # Check 4 neighbors: below, above, right, left
                # If ALL neighbors are the same color --> this pixel is interior
                neighbors = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
                is_innermost_part = True




                for neighbor_row, neighbor_col in neighbors:
                    # If neighbor is out of bounds or different color --> this is an edge pixel
                    if not (0 <= neighbor_row < rows and 0 <= neighbor_col < cols) or grid[neighbor_row, neighbor_col] != color:
                        is_innermost_part = False
                        break




                # If it is strictly interior (surrounded by same color on all 4 sides), turn it black
                if is_innermost_part:
                    output[row, col] = 0









        return output




    #---------------------------------------------
    # STRATEGY 4: Extract Colored Region (Bounding Box)
    # Find the smallest rectangle containing all non-black pixels
    #---------------------------------------------




    def crop_colored_region(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:
        # Find all positions where the pixel is not black (not 0)
        non_zero_positions = np.argwhere(grid != 0)




        # Edge case: if no colored pixels found, this doesn't apply
        if len(non_zero_positions) == 0:
            return None




        # Find the bounding box: top, bottom, left, right from my 2D array
        min_row = non_zero_positions[:, 0].min()
        max_row = non_zero_positions[:, 0].max()
        min_col = non_zero_positions[:, 1].min()
        max_col = non_zero_positions[:, 1].max()




        # Extract the bounding box region (inclusive on all sides)
        output = grid[min_row:max_row + 1, min_col:max_col + 1]









        return output




    #---------------------------------------------
    # STRATEGY 5: Color Swap (Training-based pixel mapping)
    #---------------------------------------------




    def remap_colors(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:
        # Learn the color mapping from training pairs
        train_sets = arc_problem.training_set()




        color_map = {}
        for ts in train_sets:
            train_in = ts.get_input_data().data()
            train_out = ts.get_output_data().data()




            # if same size input and output make sens
            if train_in.shape != train_out.shape:
                return None




            for r in range(train_in.shape[0]):
                for c in range(train_in.shape[1]):
                    in_val = int(train_in[r, c])
                    out_val = int(train_out[r, c])




                    # Check for consistency: if I've seen this color before,
                    # it should always map to the same output
                    if in_val in color_map and color_map[in_val] != out_val:
                        # Inconsistent mapping — this strategy might not apply cleanly
                        # But keep going, the last mapping might still work for most pixels
                        pass




                    color_map[in_val] = out_val




        # -----------Test--------------
        output = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                pixel = int(grid[r, c])
                if pixel in color_map:
                    output[r, c] = color_map[pixel]








        return output
   
    #---------------------------------------------
    # STRATEGY 6: Rotation / Flip
    #---------------------------------------------




    def rotate_or_flip (self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:




        # -------------- Training ----------------------
        # Try each transformation on training data, see which one matches




        # All my candidate transformations
        # np.rot90 k=1 is CCW, k=2 is 180, k=3 is CW
        my_transformations = [
            ("rot90_cw", lambda g: np.rot90(g, k=3)),
            ("rot90_ccw", lambda g: np.rot90(g, k=1)),
            ("rot180", lambda g: np.rot90(g, k=2)),
            ("flip_horizontal", lambda g: np.flip(g, axis=1)),
            ("flip_vertical", lambda g: np.flip(g, axis=0)),
            ("transpose", lambda g: g.T),
        ]




        winning_transform = None




        for name, transform_fn in my_transformations:
            works_on_all_training = True




            for my_ts in arc_problem.training_set():
                train_in = my_ts.get_input_data().data()
                train_out = my_ts.get_output_data().data()




                result = transform_fn(train_in)




                # Does this transformation reproduce the training output?
                if not np.array_equal(result, train_out):
                    works_on_all_training = False
                    break




            if works_on_all_training:
                winning_transform = transform_fn
                break  




        # No transformation matched training data
        if winning_transform is None:
            return None




        # -------------- Test ----------------------
        # Apply the winning transformation to my test input
        output = winning_transform(grid)




        return output
    
    # 7b6016b9.json
    def fill_any_enclosed_loop(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:

        # --- My training loop ---
        # I  need to find 3 things from training:
        # loop_color    = light blue
        # outer_fill_color = green
        # inner_fill_color = red
        inner_fill_color = None
        outer_fill_color = None
        loop_color = None
        background_base = None

        for my_ts in arc_problem.training_set():
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            if train_in.shape != train_out.shape:
                return None

            # Find most common color from my input traning which is black
            unique, counts = np.unique(train_in, return_counts=True)
            background_base  = int(unique[np.argmax(counts)])

            # Then find loop_color => light blue (7b6016b9)
            non_background_base  = [int(u) for u in unique if int(u) != background_base]
            if not non_background_base:
                return None
            loop_color = non_background_base[0]

            # Fill outer border to inner from given training output
            rows, cols = train_in.shape
            is_outside  = np.zeros((rows, cols), dtype=bool)
            queue = deque()
            for r in range(rows):
                for c in range(cols):
                    if (r == 0 or r == rows-1 or c == 0 or c == cols-1):
                        if train_in[r, c] == background_base and not is_outside[r, c]:
                            is_outside[r, c] = True
                            queue.append((r, c))
            while queue:
                r, c = queue.popleft()
                for delta_row, delta_col in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r+delta_row, c+delta_col
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not is_outside [nr, nc] and train_in[nr, nc] == background_base:
                            is_outside[nr, nc] = True
                            queue.append((nr, nc))

            # Learn outer_fill_color and inner_fill_color from training output
            # Look at what each bg pixel became in the output
            outside_result_colors  = set()
            inside_result_colors = set()
            for r in range(rows):
                for c in range(cols):
                    if train_in[r, c] == background_base:
                        out_val = int(train_out[r, c])
                        if is_outside [r, c]:
                            outside_result_colors.add(out_val)
                        else:
                            inside_result_colors.add(out_val)
        
            if len(outside_result_colors) != 1 or len(inside_result_colors) != 1:
                return None
            outer_fill_color = outside_result_colors.pop()
            inner_fill_color = inside_result_colors.pop()

            # verification of color
            for r in range(rows):
                for c in range(cols):
                    if train_in[r, c] == loop_color:
                        if train_out[r, c] != train_in[r, c]:
                            return None 
            break

        if outer_fill_color is None or inner_fill_color is None:
            return None

        # --- Test (fill) ---
    
        rows, cols = grid.shape
        is_outside = np.zeros((rows, cols), dtype=bool)
        queue = deque()

        for r in range(rows):
            for c in range(cols):
                if (r == 0 or r == rows-1 or c == 0 or c == cols-1):
                    if grid[r, c] == background_base and not is_outside[r, c]:
                        is_outside[r, c] = True
                        queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not is_outside[nr, nc] and grid[nr, nc] == background_base:
                        is_outside[nr, nc] = True
                        queue.append((nr, nc))

        # Apply: outside bg > outer_fill_color, inside bg > inner_fill_color
        test_output = grid.copy()
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == background_base:
                    test_output[r, c] = outer_fill_color if is_outside[r, c] else inner_fill_color

        return test_output
    

    # bbc9ae5d.json
    def expand_staircase(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:
       

        # --- Learn from training ---
        # We need to find:
        # 1. N           = number of colored pixels in input (top row count)
        # 2. stair_color  = what color to use 
        # 3. num_rows    = cols // 2 (learned from training)
        # 4. rule: each row adds 1 more colored pixel from left

        stair_color = None

        for my_ts in arc_problem.training_set():
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            # Input must be a single row
            if train_in.shape[0] != 1:
                return None

            # Output must have cols // 2 rows
            cols = train_in.shape[1]
            expected_rows = cols // 2
            if train_out.shape[0] != expected_rows:
                return None

            # Learn N = number of colored pixels in input row
            N = int(np.sum(train_in != 0))

            # Learn fill_color = the non-zero color in input
            non_zero = train_in[train_in != 0]
            if len(non_zero) == 0:
                return None
            stair_color = int(non_zero[0])

            # Verify the staircase rule holds in training output
            # row 0 should have N filled, row 1 has N+1, row 2 has N+2 ...
            for r in range(expected_rows):
                expected_filled = N + r    
                actual_filled = int(np.sum(train_out[r] != 0))
                if actual_filled != expected_filled:
                    return None             

            break  

        if stair_color is None:
            return None

        # --- Test data  ---

        cols = grid.shape[1]
        N = int(np.sum(grid != 0))
        num_rows = cols // 2
        non_zero = grid[grid != 0]
        if len(non_zero) == 0:
            return None
        stair_color = int(non_zero[0])
        test_output = np.zeros((num_rows, cols), dtype=int)
        for r in range(num_rows):
            filled_count = N + r 
            for c in range(filled_count):
                test_output[r, c] = stair_color  

        return test_output
    


    # 3de23699.json  
    def recolor_center_with_corner_color(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:

        # --- Learn from training ---
        # 1. corner_color = color that appears exactly 4 times (the markers)
        # 2. center_color = color that appears more than 4 times (the inner shape)
        # my rule: extract rectangle defined by corners, recolor center_color > corner_color

        corner_color = None
        center_color = None

        for my_ts in arc_problem.training_set():
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            # Find corner_color = exactly 4 pixels
            # Find center_color = more than 4 pixels
            unique, counts = np.unique(train_in, return_counts=True)

            for u, c in zip(unique, counts):
                if u == 0:
                    continue
                if c == 4:
                    corner_color = int(u)
                elif c > 4:
                    center_color = int(u)   

            if corner_color is None or center_color is None:
                return None

            break

        if corner_color is None or center_color is None:
            return None

        # --- Test might have different colors ---
        unique, counts = np.unique(grid, return_counts=True)
        test_corner_color = None
        test_center_color = None

        for u, c in zip(unique, counts):
            if u == 0:
                continue
            if c == 4:
                test_corner_color = int(u)
            elif c > 4:
                test_center_color = int(u)   

        if test_corner_color is None or test_center_color is None:
            return None

        # find the 4 corner positions using test colors
        corner_positions = np.argwhere(grid == test_corner_color)
        if len(corner_positions) != 4:
            return None

        # rectangle defined by corners
        corner_rows = corner_positions[:, 0]
        corner_cols = corner_positions[:, 1]
        rect_top    = corner_rows.min() + 1
        rect_bottom = corner_rows.max() - 1
        rect_left   = corner_cols.min() + 1
        rect_right  = corner_cols.max() - 1

        # extract inside rectangle
        rectangle_inside = grid[rect_top:rect_bottom+1, rect_left:rect_right+1]

        # recolor test_center_color >  test_corner_color
        test_output = np.zeros_like(rectangle_inside)
        for r in range(rectangle_inside.shape[0]):
            for c in range(rectangle_inside.shape[1]):
                if rectangle_inside[r, c] == test_center_color:  
                    test_output[r, c] = test_corner_color
                else:
                    test_output[r, c] = 0

        return test_output
    

    # 5c0a986e.json
    def diagonal_line_move(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:

        # Training:
        # 1. Find color1 block and color2 block
        # 2. color1 > line from top-left corner going "up-left" (-1,-1)
        # 3. color2 > line from bottom-right corner going "down right" (+1,+1)
        # 4. line stops when it hits the grid edge
        # 5. both original blocks stay unchanged

        line_color1 = None
        line_color2 = None

        for my_ts in arc_problem.training_set():
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            if train_in.shape != train_out.shape:
                return None

            # Find the two non-zero colors in input
            unique, counts = np.unique(train_in, return_counts=True)
            non_zero_colors = [int(u) for u in unique if u != 0]

            if len(non_zero_colors) != 2:
                return None

            col_1, col_2 = non_zero_colors[0], non_zero_colors[1]

            # find top-left of color1
            col1_positions = np.argwhere(train_in == col_1)
            col1_top_left_row = col1_positions[:, 0].min()
            col1_top_left_col = col1_positions[:, 1].min()

            # find bottom-right of color2
            col2_positions = np.argwhere(train_in == col_2)
            col2_bot_right_row = col2_positions[:, 0].max()
            col2_bot_right_col = col2_positions[:, 1].max() 

            # Verify color1 line goes UP-LEFT from top-left
            rows, cols = train_in.shape
            r, c = col1_top_left_row - 1, col1_top_left_col - 1
            while 0 <= r < rows and 0 <= c < cols:
                if train_out[r, c] != col_1:
                    return None
                r -= 1
                c -= 1

            # Verify color2 line goes DOWN-RIGHT from bottom-right
            r, c = col2_bot_right_row + 1, col2_bot_right_col + 1 
            while 0 <= r < rows and 0 <= c < cols:
                if train_out[r, c] != col_2:
                    return None
                r += 1
                c += 1

            line_color1 = col_1
            line_color2 = col_2
            break

        if line_color1 is None or line_color2 is None:
            return None

        # --- Test ---
        rows, cols = grid.shape
        output = grid.copy()

        # find top-left of color1 in test
        col1_positions = np.argwhere(grid == line_color1)
        if len(col1_positions) == 0:
            return None
        col1_top_left_row = col1_positions[:, 0].min()  
        col1_top_left_col = col1_positions[:, 1].min() 

        # shoot color1 line up left
        r, c = col1_top_left_row - 1, col1_top_left_col - 1  
        while 0 <= r < rows and 0 <= c < cols:
            output[r, c] = line_color1
            r -= 1
            c -= 1

        # find bottom-right of color2 in test
        col2_positions = np.argwhere(grid == line_color2)
        if len(col2_positions) == 0:
            return None
        col2_bot_right_row = col2_positions[:, 0].max()
        col2_bot_right_col = col2_positions[:, 1].max()

        # shoot color2 line down right
        r, c = col2_bot_right_row + 1, col2_bot_right_col + 1
        while 0 <= r < rows and 0 <= c < cols:
            output[r, c] = line_color2
            r += 1
            c += 1

        return output
    

    # 9af7a82c.json
    def sort_colors_left_to_right_by_count(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:

        # training -----------------------------------------------------------------------
        # count pixels of each non-zero color in input
        # sort colors by count descending most freq > leftmost
        # output rows > highest count
        # output cols = number of unique colors
        # each col fills from top with its color, rest = 0

        for my_ts in arc_problem.training_set():
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            # Count each non-zero color
            unique, counts = np.unique(train_in, return_counts=True)
            color_counts = {int(u): int(c) for u, c in zip(unique, counts) if u != 0}

            # sort by count descending
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

            # build expected output
            num_rows = max(color_counts.values())
            num_cols = len(color_counts)
            final_output = np.zeros((num_rows, num_cols), dtype=int)

            for col_idx, (color, count) in enumerate(sorted_colors):  # ← iterate sorted_colors
                for r in range(count):
                    final_output[r, col_idx] = color

            # validate against training output
            if not np.array_equal(final_output, train_out):
                return None

            break

        # --- Test  ---
        unique, counts = np.unique(grid, return_counts=True)
        color_counts = {int(u): int(c) for u, c in zip(unique, counts) if u != 0}

        # sort by count descending
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

        # build test output
        num_rows = max(color_counts.values())
        num_cols = len(color_counts)
        final_output = np.zeros((num_rows, num_cols), dtype=int)

        for col_idx, (color, count) in enumerate(sorted_colors):  
            for r in range(count):
                final_output[r, col_idx] = color

        return final_output
    

    # 22eb0ac0.json
    def connect_matching_blocks(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:

        # --- Learn from training ---
        # Rule:
        # 1. look at left edge (col 0) and right edge (last col) of each row
        # 2. if left color == right color AND both non-zero > fill entire row with that color
        # 3. if left color != right color > leave row unchanged

        for my_ts in arc_problem.training_set():
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            if train_in.shape != train_out.shape:
                return None

            rows, cols = train_in.shape

            # check where left and right cols match
            for r in range(rows):
                left_block  = int(train_in[r, 0])       
                right_block = int(train_in[r, cols - 1]) 

                if left_block != 0 and right_block != 0 and left_block == right_block:
                    expected_output_row = np.full(cols, left_block, dtype=int)
                    if not np.array_equal(train_out[r], expected_output_row):
                        return None  
                else:
                    if not np.array_equal(train_out[r], train_in[r]):
                        return None  
            break 

        # ------- Apply to test grid  ------- 
        rows, cols = grid.shape
        output = grid.copy()  

        for r in range(rows):
            left_block  = int(grid[r, 0])        
            right_block = int(grid[r, cols - 1])  

           
            if left_block != 0 and right_block != 0 and left_block == right_block:
                output[r, :] = left_block   

        return output
    
    
    # 25d487eb.json
    def lone_color_extension(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:

        # --- Learn from training ---
        # find lone color (1 pixel) and non_lone_color (many pixels)
        # lone color sits on edge of non_lone shape bounding box
        # trail extends from lone color outward to grid edge

        for my_ts in arc_problem.training_set():
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            if train_in.shape != train_out.shape:
                return None

            # Find lone_color = exactly 1 pixel
            # Find non_lone_color = more than 1 pixel
            unique, counts = np.unique(train_in, return_counts=True)
            lone_color = None
            non_lone_color = None

            for u, c in zip(unique, counts):
                if u == 0: continue
                if c == 1:
                    lone_color = int(u)
                elif c > 1:
                    non_lone_color = int(u)

            if lone_color is None or non_lone_color is None:
                return None

            # Find position of lone color pixel
            lone_pos = np.argwhere(train_in == lone_color)[0]
            lone_r, lone_c = int(lone_pos[0]), int(lone_pos[1])

            # Find bounding box of non_lone shape
            shape_positions = np.argwhere(train_in == non_lone_color)
            shape_min_r = shape_positions[:, 0].min()
            shape_max_r = shape_positions[:, 0].max()
            shape_min_c = shape_positions[:, 1].min()
            shape_max_c = shape_positions[:, 1].max()

            # Determine direction: which edge is lone color on?
            if lone_c == shape_min_c:
                delta_row, delta_col = 0, +1    
            elif lone_c == shape_max_c:
                delta_row, delta_col = 0, -1    
            elif lone_r == shape_min_r:
                delta_row, delta_col = +1, 0   
            elif lone_r == shape_max_r:
                delta_row, delta_col = -1, 0   
            else:
                return None

            # Verify trail exists in training output
            # walk from lone color outward — every pixel must be lone_color

            rows, cols = train_in.shape
            r, c = lone_r + delta_row, lone_c + delta_col
            while 0 <= r < rows and 0 <= c < cols:
                if train_in[r, c] == 0:  
                    if train_out[r, c] != lone_color:
                        return None
                r += delta_row
                c += delta_col

            break  # learned from first training pair

        # --- Test ---

        rows, cols = grid.shape
        output = grid.copy() 

        # Fins colors from test grid
        unique, counts = np.unique(grid, return_counts=True)
        lone_color = None
        non_lone_color = None

        for u, c in zip(unique, counts):
            if u == 0: continue
            if c == 1:
                lone_color = int(u)
            elif c > 1:
                non_lone_color = int(u)

        if lone_color is None or non_lone_color is None:
            return None

        # Find lone color position in test
        lone_pos = np.argwhere(grid == lone_color)[0]
        lone_r, lone_c = int(lone_pos[0]), int(lone_pos[1])  
        # Find shape bounding box in test
        shape_positions = np.argwhere(grid == non_lone_color)  
        shape_min_r = shape_positions[:, 0].min()
        shape_max_r = shape_positions[:, 0].max()
        shape_min_c = shape_positions[:, 1].min()
        shape_max_c = shape_positions[:, 1].max()

        # Determine direction from lone color edge position in test
        if lone_c == shape_min_c:
            delta_row, delta_col = 0, +1   
        elif lone_c == shape_max_c:
            delta_row, delta_col = 0, -1    
        elif lone_r == shape_min_r:
            delta_row, delta_col = +1, 0   
        elif lone_r == shape_max_r:
            delta_row, delta_col = -1, 0    
        else:
            return None

      


        r, c = lone_r + delta_row, lone_c + delta_col
        while 0 <= r < rows and 0 <= c < cols:
            if grid[r, c] == 0:         
                output[r, c] = lone_color
            r += delta_row
            c += delta_col

        return output
    
    # 62c24649.json 
    def mirror_my_input(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:

        # ===============  training ===================================================================
        # My rule
        # n*m to 2(n*m)
        # top-left     = original input
        # top-right    = input flipped horizontally (mirror left-right)
        # bottom-left  = input flipped vertically   (mirror top-bottom)
        # bottom-right = input flipped both ways

        for my_ts in arc_problem.training_set():
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            rows, cols = train_in.shape

            # Output must be exactly double by tying 4 piece together
            if train_out.shape != (rows * 2, cols * 2):
                return None

            # Build expected output from 4 quadrants, as it is, horizonal flip, vertical flip, vertical right
            output_top_left     = train_in                           
            output_top_right    = np.flip(train_in, axis=1)          
            output_bottom_left  = np.flip(train_in, axis=0)          
            output_bottom_right = np.flip(np.flip(train_in, axis=1), axis=0)  

            final_output = np.block([
                [output_top_left, output_top_right],
                [output_bottom_left, output_bottom_right]
            ])

            # Verify against training output
            if not np.array_equal(final_output, train_out):
                return None

            break 

        # --- Apply to test grid ---
        output_top_left     = grid
        output_top_right    = np.flip(grid, axis=1)
        output_bottom_left  = np.flip(grid, axis=0)
        output_bottom_right = np.flip(np.flip(grid, axis=1), axis=0)  


        output = np.block([
            [output_top_left,    output_top_right],
            [output_bottom_left, output_bottom_right]
        ])

        return output
    
    # 74dd1130.json done by above


    # 3428a4f5.json
    def split_and_recolor_using_xor(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:

        # top      xor   output
        # separator => 
        # bottom
        # if exactly one side has input color => output recolor
        # if both same => output black

        
        # my_input_color will map to my_recolor_val
        my_recolor_val = None 
        my_input_color = None 

        for my_ts in arc_problem.training_set():
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            rows, cols = train_in.shape

            # find my separator row that is same color and not just black pixel, and nowhere in grid
            sep_row = None
            sep_color = None
            for r in range(rows):
                unique_vals = np.unique(train_in[r, :])
                if len(unique_vals) == 1 and unique_vals[0] != 0:
                    separator_color = int(unique_vals[0])
                    # is this only my color that is unque in this matrix?
                    other_similar_rows = False
                    for other_rows in range(rows):
                        if other_rows == r:
                            continue
                        if separator_color in train_in[other_rows, :]:
                            other_similar_rows = True
                            break
                    if not other_similar_rows:
                        sep_row = r
                        sep_color = separator_color
                        break

            # if sep_row is None:
            #     return None

            # now lets split into top and bottom halves and check if output grid matches the top half grid size
            top_half = train_in[:sep_row, :]
            bottom_half = train_in[sep_row+1:, :]

            # if top_half.shape != bottom_half.shape or top_half.shape != train_out.shape:
            #     return None

            # lets now learn the input color so I can compare xor logic
            unique, counts = np.unique(top_half, return_counts=True)
            my_input_color = None
            for idx in np.argsort(counts)[::-1]:
                if int(unique[idx]) != 0 and int(unique[idx]) != sep_color:
                    my_input_color = int(unique[idx])
                    break

            if my_input_color is None:
                return None

            # learning the output value so I can map
            nonzero_out  = train_out[train_out != 0]
            if len(nonzero_out) == 0:
                return None
            my_recolor_val  = int(nonzero_out[0])

            # verify XOR rule holds on all pixels in training. I do this by comparing top and bottom half and see if input color match
            for r in range(top_half.shape[0]):
                for c in range(cols):
                    top_has = (top_half[r, c] == my_input_color)   
                    bot_has = (bottom_half[r, c] == my_input_color)   
                    xor = top_has ^ bot_has                 
                    new_color = my_recolor_val  if xor else 0
                    if train_out[r, c] != new_color:
                        return None  

            break 

        if my_recolor_val is None or my_input_color is None:
            return None

        # --- Apply to test grid ---------------------------------------------------
        rows, cols = grid.shape

        # same logic for setup as train
        sep_row = None
        sep_color = None
        for r in range(rows):
            unique_vals = np.unique(grid[r, :])
            if len(unique_vals) == 1 and unique_vals[0] != 0:
                separator_color = int(unique_vals[0])
           
                other_similar_rows = False
                for other_rows in range(rows):
                    if other_rows == r:
                        continue
                    if separator_color in grid[other_rows, :]:
                        other_similar_rows = True
                        break
                if not other_similar_rows:
                    sep_row = r
                    sep_color = separator_color
                    break

        if sep_row is None:
            return None

        # Split test into top and bottom
        top_half = grid[:sep_row, :]
        bottom_half = grid[sep_row + 1:, :]

        if top_half.shape != bottom_half.shape:
            return None

        # Apply XOR rule using learned input_grid_color and mapped value
        output = np.zeros_like(top_half)
        for r in range(top_half.shape[0]):
            for c in range(cols):
                top_has = (top_half[r, c] == my_input_color)   
                bot_has = (bottom_half[r, c] == my_input_color)  
                xor = top_has ^ bot_has                
                output[r, c] = my_recolor_val if xor else 0

        return output
    

    #---------------------- Milestone D -----------------------------------
    
    def fill_closed_regions_with_hint_majority(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray:
        """
        Strategy for tasks like 4b6b68e5:
            - find large connected single-color components that act as closed outlines
            - for each enclosed area, fill with the most frequent non-zero hint color inside
            - erase stray non-outline/hint pixels outside filled enclosed regions
        """
        rows, cols = grid.shape
        my_direction = ((1, 0), (-1, 0), (0, 1), (0, -1))

        def components_of_color(color: int) -> list[list[tuple[int, int]]]:
            visited = np.zeros((rows, cols), dtype=bool)
            all_islands_of_this_color: list[list[tuple[int, int]]] = []
            for r in range(rows):
                for c in range(cols):
                    if visited[r, c] or int(grid[r, c]) != color:
                        continue
                    bfs_q = deque([(r, c)])
                    visited[r, c] = True
                    comp: list[tuple[int, int]] = []
                    while bfs_q:
                        curr_row, curr_col = bfs_q.popleft()
                        comp.append((curr_row, curr_col))
                        for delta_row, delta_col in my_direction:
                            neighbor_row, neighbor_col = curr_row + delta_row, curr_col + delta_col
                            if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols and not visited[neighbor_row, neighbor_col] and int(grid[neighbor_row, neighbor_col]) == color:
                                visited[neighbor_row, neighbor_col] = True
                                bfs_q.append((neighbor_row, neighbor_col))
                    all_islands_of_this_color.append(comp)
            return all_islands_of_this_color

        def enclosed_cells(boundary_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
            # these are my outside pixels
            is_wall = np.zeros((rows, cols), dtype=bool)
            for r, c in boundary_cells:
                is_wall[r, c] = True

            is_outside = np.zeros((rows, cols), dtype=bool)
            bfs_q = deque()

            for r in range(rows):
                for c in range(cols):
                    is_on_border = (r == 0 or r == rows-1 or c == 0 or c == cols-1)
                    if is_on_border and not is_wall[r, c]:
                        is_outside[r, c] = True
                        bfs_q.append((r, c))

            while bfs_q:
                curr_row, curr_col = bfs_q.popleft()
                for delta_row, delta_col in my_direction:
                    neighbor_row = curr_row + delta_row
                    neighbor_col = curr_col + delta_col
                    in_bounds = 0 <= neighbor_row < rows and 0 <= neighbor_col < cols
                    not_wall = not is_wall[neighbor_row, neighbor_col] if in_bounds else False
                    not_yet_outside = not is_outside[neighbor_row, neighbor_col] if in_bounds else False
                    if in_bounds and not_wall and not_yet_outside:
                        is_outside[neighbor_row, neighbor_col] = True
                        bfs_q.append((neighbor_row, neighbor_col))

            enclosed_list: list[tuple[int, int]] = []
            for r in range(rows):
                for c in range(cols):
                    if not is_wall[r, c] and not is_outside[r, c]:
                        enclosed_list.append((r, c))
            return enclosed_list

        output = grid.copy()

        
        pixel_should_be_kept = np.zeros((rows, cols), dtype=bool)

        
        for outline_color in map(int, np.unique(grid)):
            if outline_color == 0:
                continue  
            for island_pixels in components_of_color(outline_color):

                is_substantial_outline = len(island_pixels) >= 6
                if not is_substantial_outline:
                    continue

                # mark all outline pixels as keep
                for r, c in island_pixels:
                    pixel_should_be_kept[r, c] = True

                # find pixels enclosed inside this outline shape
                interior_pixels = enclosed_cells(island_pixels)
                if not interior_pixels:
                    continue  


                hint_color_counts: dict[int, int] = {}
                for r, c in interior_pixels:
                    pixel_value = int(grid[r, c])
                    if pixel_value != 0:
                        hint_color_counts[pixel_value] = hint_color_counts.get(pixel_value, 0) + 1

                if not hint_color_counts:
                    continue  


                # fill color = the most frequent hint color inside
                majority_fill_color = max(hint_color_counts, key=hint_color_counts.get)

                # fill entire interior with majority color and mark as keep
                for r, c in interior_pixels:
                    output[r, c] = majority_fill_color
                    pixel_should_be_kept[r, c] = True
        # zero out any pixel not marked as my hints
        for r in range(rows):
            for c in range(cols):
                pixel_is_nonzero = int(output[r, c]) != 0
                if pixel_is_nonzero and not pixel_should_be_kept[r, c]:
                    output[r, c] = 0

        return output
    
    def solve_comparing_xor_halves(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        
        31d5ba1a
        - Input is two equal-height blocks stacked vertically.
        - Treat non-zero cells in the top and bottom block as binary masks.
        - Output is the XOR of top and bottom using the learned output color.
        
        """
        rows, cols = grid.shape
        if rows % 2 != 0:
            return None

        half = rows // 2
        top_block = grid[:half, :]
        bottom_block = grid[half:, :]

        # one side filled => keep
        top_non_zero = top_block != 0
        bottom_non_zero = bottom_block != 0
        xor_val = np.logical_xor(top_non_zero, bottom_non_zero)

        # learn output color from training outputs
        output_color = None
        all_output_colors = set()
        for my_ts in arc_problem.training_set():
            train_out = my_ts.get_output_data().data()
            for c in np.unique(train_out):
                if int(c) != 0:
                    all_output_colors.add(int(c))

        if len(all_output_colors) != 1:
            return None
        output_color = next(iter(all_output_colors))

        #  -------------Now apply to test grid------------------------
        output = np.zeros((half, cols), dtype=int)
        output[xor_val] = output_color
        return output
        
    def solve_connecting_crosses(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve pattern like 60a26a3e.
        """
        rows, cols = grid.shape

        # learn the connector color from training
        learned_line_colors = set()
        for my_ts in arc_problem.training_set():
            train_in  = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            in_colors  = {int(color) for color in np.unique(train_in)  if int(color) != 0}
            out_colors = {int(color) for color in np.unique(train_out) if int(color) != 0}

            added_colors = out_colors - in_colors
            if len(added_colors) != 1:
                return None

            learned_line_colors |= added_colors

        if len(learned_line_colors) != 1:
            return None

        color_of_line = next(iter(learned_line_colors))

        # confirm the rule holds on training
        for my_ts in arc_problem.training_set():
            train_in  = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()
            train_rows, train_cols = train_in.shape

            # find all cross centers — safe bounds check on all 4 neighbors
            centers_by_color = {}
            for row in range(train_rows):
                for col in range(train_cols):
                    if int(train_in[row, col]) != 0:
                        continue

                    has_up    = (row > 0)              and int(train_in[row-1, col]) != 0
                    has_down  = (row < train_rows - 1) and int(train_in[row+1, col]) != 0
                    has_left  = (col > 0)              and int(train_in[row, col-1]) != 0
                    has_right = (col < train_cols - 1) and int(train_in[row, col+1]) != 0

                    if not (has_up and has_down and has_left and has_right):
                        continue

                    up    = int(train_in[row-1, col])
                    down  = int(train_in[row+1, col])
                    left  = int(train_in[row, col-1])
                    right = int(train_in[row, col+1])

                    if up == down == left == right:
                        if up not in centers_by_color:
                            centers_by_color[up] = []
                        centers_by_color[up].append((row, col))

            if not centers_by_color:
                return None

            matched_training_rule = False

            for color_of_cross, center_of_cross in centers_by_color.items():
                training_output = train_in.copy()

                # connect crosses on same row horizontally
                centers_by_row = {}
                for center_row, center_col in center_of_cross:
                    if center_row not in centers_by_row:
                        centers_by_row[center_row] = []
                    centers_by_row[center_row].append(center_col)

                for same_row, col_positions in centers_by_row.items():
                    col_positions.sort()
                    for i in range(len(col_positions) - 1):
                        left_center_col  = col_positions[i]
                        right_center_col = col_positions[i + 1]

                        can_connect = True
                        for col in range(left_center_col + 2, right_center_col - 1):
                            if int(train_in[same_row, col]) != 0:
                                can_connect = False
                                break

                        if can_connect:
                            for col in range(left_center_col + 2, right_center_col - 1):
                                training_output[same_row, col] = color_of_line

                # connect crosses on same col vertically
                centers_by_col = {}
                for center_row, center_col in center_of_cross:
                    if center_col not in centers_by_col:
                        centers_by_col[center_col] = []
                    centers_by_col[center_col].append(center_row)

                for same_col, row_positions in centers_by_col.items():
                    row_positions.sort()
                    for i in range(len(row_positions) - 1):
                        top_center_row    = row_positions[i]
                        bottom_center_row = row_positions[i + 1]

                        can_connect = True
                        for row in range(top_center_row + 2, bottom_center_row - 1):
                            if int(train_in[row, same_col]) != 0:
                                can_connect = False
                                break

                        if can_connect:
                            for row in range(top_center_row + 2, bottom_center_row - 1):
                                training_output[row, same_col] = color_of_line

                if np.array_equal(training_output, train_out):
                    matched_training_rule = True
                    break

            if not matched_training_rule:
                return None

        # apply to test grid
        centers_by_color = {}
        for row in range(rows):
            for col in range(cols):
                if int(grid[row, col]) != 0:
                    continue

                has_up    = (row > 0)        and int(grid[row-1, col]) != 0
                has_down  = (row < rows - 1) and int(grid[row+1, col]) != 0
                has_left  = (col > 0)        and int(grid[row, col-1]) != 0
                has_right = (col < cols - 1) and int(grid[row, col+1]) != 0

                if not (has_up and has_down and has_left and has_right):
                    continue

                up    = int(grid[row-1, col])
                down  = int(grid[row+1, col])
                left  = int(grid[row, col-1])
                right = int(grid[row, col+1])

                if up == down == left == right:
                    if up not in centers_by_color:
                        centers_by_color[up] = []
                    centers_by_color[up].append((row, col))

        if not centers_by_color:
            return None

        best_output      = None
        best_added_count = -1

        for color_of_cross, center_of_cross in centers_by_color.items():
            output      = grid.copy()
            added_count = 0

            # connect crosses on same row horizontally
            centers_by_row = {}
            for center_row, center_col in center_of_cross:
                if center_row not in centers_by_row:
                    centers_by_row[center_row] = []
                centers_by_row[center_row].append(center_col)

            for same_row, col_positions in centers_by_row.items():
                col_positions.sort()
                for i in range(len(col_positions) - 1):
                    left_center_col  = col_positions[i]
                    right_center_col = col_positions[i + 1]

                    can_connect = True
                    for col in range(left_center_col + 2, right_center_col - 1):
                        if int(grid[same_row, col]) != 0:
                            can_connect = False
                            break

                    if can_connect:
                        for col in range(left_center_col + 2, right_center_col - 1):
                            if int(output[same_row, col]) == 0:
                                output[same_row, col] = color_of_line
                                added_count += 1

            # connect crosses on same col vertically
            centers_by_col = {}
            for center_row, center_col in center_of_cross:
                if center_col not in centers_by_col:
                    centers_by_col[center_col] = []
                centers_by_col[center_col].append(center_row)

            for same_col, row_positions in centers_by_col.items():
                row_positions.sort()
                for i in range(len(row_positions) - 1):
                    top_center_row    = row_positions[i]
                    bottom_center_row = row_positions[i + 1]

                    can_connect = True
                    for row in range(top_center_row + 2, bottom_center_row - 1):
                        if int(grid[row, same_col]) != 0:
                            can_connect = False
                            break

                    if can_connect:
                        for row in range(top_center_row + 2, bottom_center_row - 1):
                            if int(output[row, same_col]) == 0:
                                output[row, same_col] = color_of_line
                                added_count += 1

            if added_count > best_added_count:
                best_added_count = added_count
                best_output      = output

        if best_output is None or best_added_count <= 0:
            return None

        return best_output
            
        
   
    


    def solve_histogram_from_frequent_patch(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Solve pattern like 81c0276b:
        - Detect separator color that acts like a boxed wall.
        - Partition into cell blocks between boxes.
        - Count non-separator colors appearing in cells.
        - Build colored hist based on patch of color
        
        """
        rows, cols = grid.shape
        

        all_row_colors = []
        for r in range(rows):
            first_pixel = int(grid[r, 0])
            if first_pixel == 0:
                continue  
            entire_row_is_same_color = np.all(grid[r, :] == first_pixel)
            if entire_row_is_same_color:
                all_row_colors.append(first_pixel)


        all_col_colors = []
        for c in range(cols):
            first_pixel = int(grid[0, c])
            if first_pixel == 0:
                continue 
            entire_col_is_same_color = np.all(grid[:, c] == first_pixel)
            if entire_col_is_same_color:
                all_col_colors.append(first_pixel)

        row_color_set = set(all_row_colors)
        col_color_set = set(all_col_colors)
        row_col_intersect_entity = row_color_set & col_color_set  


        if len(row_col_intersect_entity) != 1:
            return None

        my_separator = next(iter(row_col_intersect_entity))
    
        # add -1 at start and rows/cols at end so every cell block has a start and end
        rows_to_look_in = [r for r in range(rows) if np.all(grid[r, :] == my_separator)]
        cols_to_look_in = [c for c in range(cols) if np.all(grid[:, c] == my_separator)]

        row_cuts = [-1] + rows_to_look_in + [rows]
        col_cuts = [-1] + cols_to_look_in + [cols]

        

        # this is looking between dividers
        color_counts = {}  

        for ri in range(len(row_cuts) - 1):
            
            row_start = row_cuts[ri] + 1   
            row_end   = row_cuts[ri + 1]
            
            if row_start >= row_end:
                continue 
            for ci in range(len(col_cuts) - 1):
            
                col_start = col_cuts[ci] + 1  
                col_end   = col_cuts[ci + 1]
                
                if col_start >= col_end:
                    continue 
                
            
                cell_block = grid[row_start:row_end, col_start:col_end]
                
                
                blob_colors = []
                for v in np.unique(cell_block):
                    if int(v) != 0 and int(v) != my_separator:
                        blob_colors.append(int(v))
                
                if not blob_colors:
                    continue  
                
                if len(blob_colors) != 1:
                    return None 
                
                # count this color
                found_color = blob_colors[0]
                if found_color not in color_counts:
                    color_counts[found_color] = 0
                color_counts[found_color] += 1

        

        if not color_counts:
            return None
        
        
        # sort by count first, then by color value (for tie-breaking)
        items = sorted(color_counts.items(), key=lambda kv: (kv[1], kv[0]))
        out_height = len(items)
        out_width = max(cnt for _, cnt in items)
        output = np.zeros((out_height, out_width), dtype=int)

        for row_idx, (color, count) in enumerate(items):
            for c in range(count):
                output[row_idx, c] = color

        return output
    
    def solve_inner_fill_with_reflection_in_frames(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        18419cfa:
        - frame color = most frequent non-zero color (blue border shapes)
        - fill color  = less frequent non-zero color (red pattern inside)
        - for each frame island, reflect fill pixels to all 4 symmetric positions
        """

        # find all non-zero colors — need at least 2 (frame + fill)
        all_nonzero_colors = []
        for c in np.unique(grid):
            if int(c) != 0:
                all_nonzero_colors.append(int(c))
        if len(all_nonzero_colors) < 2:
            return None

        # frame color = color with most pixels
        pixel_count_per_color = {}
        for color in all_nonzero_colors:
            pixel_count_per_color[color] = int(np.sum(grid == color))

        outer_border_color = None
        max_count = -1
        for color, count in pixel_count_per_color.items():
            if count > max_count:
                max_count = count
                outer_border_color = color

        # fill color = everything that is not the frame color
        inside_fill_candidates = []
        for color in all_nonzero_colors:
            if color != outer_border_color:
                inside_fill_candidates.append(color)

        if len(inside_fill_candidates) != 1:
            return None
        inside_fill_color = inside_fill_candidates[0]

        # --- BFS to find each frame island ---
        rows, cols = grid.shape
        visited = np.zeros((rows, cols), dtype=bool)
        my_direction = ((1, 0), (-1, 0), (0, 1), (0, -1))
        output = grid.copy()

        for r in range(rows):
            for c in range(cols):
                if visited[r, c] or int(grid[r, c]) != outer_border_color:
                    continue

                # BFS — find all connected border pixels of this frame island
                bfs_q = deque([(r, c)])
                visited[r, c] = True
                current_island = []
                while bfs_q:
                    curr_row, curr_col = bfs_q.popleft()
                    current_island.append((curr_row, curr_col))
                    for delta_row, delta_col in my_direction:
                        neighbor_row = curr_row + delta_row
                        neighbor_col = curr_col + delta_col
                        in_bounds = 0 <= neighbor_row < rows and 0 <= neighbor_col < cols
                        not_visited = not visited[neighbor_row, neighbor_col] if in_bounds else False
                        is_border = int(grid[neighbor_row, neighbor_col]) == outer_border_color if in_bounds else False
                        if in_bounds and not_visited and is_border:
                            visited[neighbor_row, neighbor_col] = True
                            bfs_q.append((neighbor_row, neighbor_col))

               
                if len(current_island) < 10:
                    continue

                # find bounding box of this frame island
                top_row    = min(rr for rr, _ in current_island)
                bottom_row = max(rr for rr, _ in current_island)
                left_col   = min(cc for _, cc in current_island)
                right_col  = max(cc for _, cc in current_island)

                # reflect fill pixels to all 4 symmetric positions inside bounding box
                for rr in range(top_row, bottom_row + 1):
                    for cc in range(left_col, right_col + 1):

                        # only reflect fill color pixels
                        if int(grid[rr, cc]) != inside_fill_color:
                            continue

                        # calculate mirror positions across center of bounding box
                        rr_mirror = top_row + bottom_row - rr  
                        cc_mirror = left_col + right_col - cc   

                        # fill all 4 symmetric positions
                        for target_row, target_col in [(rr, cc),
                                                    (rr_mirror, cc),
                                                    (rr, cc_mirror),
                                                    (rr_mirror, cc_mirror)]:
                            is_inside_box = top_row <= target_row <= bottom_row and left_col <= target_col <= right_col
                            is_not_border = int(output[target_row, target_col]) != outer_border_color
                            if is_inside_box and is_not_border:
                                output[target_row, target_col] = inside_fill_color

        return output
     
     
    def overlay_on_empty(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        Split at middle separator column, then overlay right onto left's empty cells.
        """
        rows, cols = grid.shape
        
        # Find separator column that are all same and not 0
        sep_cols = [c for c in range(cols) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != 0]
        mid_candidates = [c for c in sep_cols if c == (cols - 1 - c)]
        if len(mid_candidates) != 1:
            return None
        sep = mid_candidates[0]
        
        # Split left and right blocks
        left = grid[:, :sep]
        right = grid[:, sep + 1:]
        if left.shape != right.shape:
            return None
        
        # Check if colors overlap
        overlap = np.any((left != 0) & (right != 0))
        if overlap:
            return left.copy()
        
        # Overlay right onto left's empty cells
        result = left.copy()
        mask = (result == 0) & (right != 0)
        result[mask] = right[mask]
        return result

    
    def solve_with_or_halves(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        195ba7dc:
        - find center separator column (single color, exact middle)
        - split into left and right halves
        - OR the two masks => wherever either side has a pixel => output has pixel
        - output color learned from training
        """
        rows, cols = grid.shape

        # --- training verification ---
        output_color = None
        for my_ts in arc_problem.training_set():
            train_in  = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            train_rows, train_cols = train_in.shape

            # find separator column — must be exact center, all same non-zero color
            separator_col = None
            for c in range(train_cols):
                is_center    = (c == train_cols - 1 - c)
                all_same     = len(set(int(v) for v in train_in[:, c])) == 1
                is_not_black = int(train_in[0, c]) != 0
                if is_center and all_same and is_not_black:
                    separator_col = c
                    break

            if separator_col is None:
                return None

            # split into left and right halves
            left_half  = train_in[:, :separator_col]
            right_half = train_in[:, separator_col + 1:]

            if left_half.shape[1] != right_half.shape[1]:
                return None

            # OR the two halves — wherever either side has a pixel → keep it
            left_block   = left_half != 0
            right_block  = right_half != 0
            or_result    = np.logical_or(left_block, right_block)

            # learn output color — single non-zero color in training output
            for c in np.unique(train_out):
                if int(c) != 0:
                    if output_color is not None and output_color != int(c):
                        return None
                    output_color = int(c)

            if output_color is None:
                return None

            # verify OR rule produces exact match with training output
            expected_out = np.zeros_like(left_half, dtype=int)
            expected_out[or_result] = output_color
            if not np.array_equal(expected_out, train_out):
                return None

        if output_color is None:
            return None

        # --- apply to test grid ---
        # find separator column in test grid
        separator_col = None
        for c in range(cols):
            is_center    = (c == cols - 1 - c)
            all_same     = len(set(int(v) for v in grid[:, c])) == 1
            is_not_black = int(grid[0, c]) != 0
            if is_center and all_same and is_not_black:
                separator_col = c
                break

        if separator_col is None:
            return None

        # split into left and right halves
        left_half  = grid[:, :separator_col]
        right_half = grid[:, separator_col + 1:]

        if left_half.shape[1] != right_half.shape[1]:
            return None

        # OR the two halves
        left_block  = left_half != 0
        right_block = right_half != 0
        or_result   = np.logical_or(left_block, right_block)

        # fill with output color
        output = np.zeros_like(left_half, dtype=int)
        output[or_result] = output_color
        return output
        
        
    
    
    
    
    ## ---------------update-----
    
    
    
    
    
    
    # to review
    def generate_v_with_red_and_blue_diagonals(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        c1990cce:
        - input is 1 row with a single red (2) pixel
        - output is a square grid of size = input width
        - red forms a V shape centered on red_col
        - blue fills diagonal lines below the V, pattern based on (col-red_col) % 4
        """
        rows, cols = grid.shape
        if rows != 1:
            return None

        # find the single red pixel — its column is the center
        red_dot = np.argwhere(grid == 2)
        if len(red_dot) != 1:
            return None
        red_col = int(red_dot[0][1])

        size = cols
        output = np.zeros((size, size), dtype=int)

        # draw red V: arms spread outward from red_col as rows increase
        for row in range(size):
            left_col  = red_col - row
            right_col = red_col + row
            if 0 <= left_col  < size:
                output[row, left_col]  = 2
            if 0 <= right_col < size and right_col != left_col:
                output[row, right_col] = 2

        # draw blue diagonals below V
        # rule: blue at (row, col) if (col - red_col) % 4 == (row - red_col) % 4
        # and the cell is empty (not already red)
        for row in range(size):
            target_remainder = (row - red_col) % 4
            for col in range(size):
                if output[row, col] != 0:
                    continue  # skip red pixels
                if (col - red_col) % 4 == target_remainder:
                    # only place blue outside the V arms
                    left_arm  = red_col - row
                    right_arm = red_col + row
                    if col < left_arm or col > right_arm:
                        output[row, col] = 1

        return output
    
    # need to review
    def output_from_multiple_flipped_blocks(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        
        c48954c1-like tasks.

        """
        # The square shape is what I am looking for
        rows, cols = grid.shape
        if rows != cols or rows == 0:
            return None
        # first I start with 180 deg rotation
        my_start_block = np.rot90(grid, 2)
        left_right_flip = np.fliplr(my_start_block)
        up_down_flip = np.flipud(my_start_block)
        both_flips = np.flipud(left_right_flip)
         
       
        # when i carrefully look it is a combination of  
        output = np.block([
            [my_start_block, left_right_flip, my_start_block],
            [up_down_flip, both_flips, up_down_flip],
            [my_start_block, left_right_flip, my_start_block]
        ])
        return output
    
    # to review
    def fit_colored_blocks_into_empty_slots(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        67c52801
 
        """
        rows, cols = grid.shape
        if rows < 2:
            return None

        # The last row is which is my stable base
        stable_row = grid[rows - 1, :]
        if not np.all(stable_row == stable_row[0]) or int(stable_row[0]) == 0:
            return None
        base_color = int(stable_row[0])

        # The row above the base tells me where the support posts and empty slots are.
        is_fitting_row = grid[rows - 2, :]

        # Find every empty slot in the fitting row.
        empty_slots = []
        col = 0
        while col < cols:
            if is_fitting_row[col] != 0:
                col += 1
                continue

            start_col = col
            while col < cols and is_fitting_row[col] == 0:
                col += 1
            end_col = col - 1
            empty_slots.append((start_col, end_col))

        if not empty_slots:
            return None

        # Count how many cells belong to each non-black, non-base color above the base row.
        color_counts = {}
        for row in range(rows - 1):
            for col in range(cols):
                color = int(grid[row, col])
                if color == 0 or color == base_color:
                    continue
                color_counts[color] = color_counts.get(color, 0) + 1

        # I expect one color group for each empty slot.
        if len(color_counts) != len(empty_slots):
            return None

        # Sort slots by width and sort colors by number of cells.
        # That way the smaller color block goes into the smaller slot.
        sorted_slots = sorted(empty_slots, key=lambda slot: (slot[1] - slot[0] + 1, slot[0]))
        sorted_colors = sorted(color_counts.items(), key=lambda item: item[1])

        # Start building the output with only the base and fitting posts.
        output = np.zeros_like(grid)
        output[rows - 1, :] = base_color
        output[rows - 2, is_fitting_row != 0] = base_color

        # Pack each color into its matching slot as a rectangle.
        for (start_col, end_col), (color, count) in zip(sorted_slots, sorted_colors):
            slot_width = end_col - start_col + 1

            # The color area must fit perfectly into a rectangle of this width.
            if count % slot_width != 0:
                return None

            block_height = count // slot_width
            top_row = rows - 2 - block_height + 1
            if top_row < 0:
                return None

            output[top_row:rows - 1, start_col:end_col + 1] = color

        return output
    
    
    
    def extract_the_inside_box_and_recolor_it(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        e9b4f6fc:
        - find the largest solid rectangular panel
        - read color mapping pairs outside the panel (try both directions)
        - recolor panel using the mapping
        """

        def find_solid_panels(arr):
            # find all connected islands and keep only solid rectangles (fills entire bounding box)
            rows, cols = arr.shape
            seen = np.zeros_like(arr, dtype=bool)
            solid_panels = []

            colored_block = np.argwhere(arr != 0)
            for start_row, start_col in colored_block:
                start_row, start_col = int(start_row), int(start_col)
                if seen[start_row, start_col]:
                    continue

                queue = deque([(start_row, start_col)])
                seen[start_row, start_col] = True
                current_block = []

                while queue:
                    row, col = queue.popleft()
                    current_block.append((row, col))

                    for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        next_row = row + row_step
                        next_col = col + col_step
                        if 0 <= next_row < rows and 0 <= next_col < cols:
                            if not seen[next_row, next_col] and arr[next_row, next_col] != 0:
                                seen[next_row, next_col] = True
                                queue.append((next_row, next_col))

                # check if solid rectangle
                extracted_rows = [r for r, _ in current_block]
                extracted_cols = [c for _, c in current_block]
                top_row    = min(extracted_rows)
                bottom_row = max(extracted_rows)
                left_col   = min(extracted_cols)
                right_col  = max(extracted_cols)
                height = bottom_row - top_row + 1
                width  = right_col  - left_col  + 1

                if len(current_block) == height * width:
                    solid_panels.append((top_row, bottom_row, left_col, right_col))

            # sort biggest first
            solid_panels.sort(key=lambda box: -((box[1]-box[0]+1) * (box[3]-box[2]+1)))
            return solid_panels

        def build_mapping_and_output(arr, top_row, bottom_row, left_col, right_col, panel_is_on_right):
            rows, cols = arr.shape
            main_extract = arr[top_row:bottom_row+1, left_col:right_col+1].copy()
            extracted_colors = {int(value) for value in np.unique(main_extract)}

            mapping = {}
            for row in range(rows):
                for col in range(cols - 1):
                    # skip pairs inside the main box
                    if top_row <= row <= bottom_row and left_col <= col <= right_col:
                        continue
                    if top_row <= row <= bottom_row and left_col <= col + 1 <= right_col:
                        continue

                    left_color  = int(arr[row, col])
                    right_color = int(arr[row, col + 1])

                    if left_color == 0 or right_color == 0 or left_color == right_color:
                        continue

                    if panel_is_on_right:
                        # pair: [new_color][panel_color]
                        if right_color in extracted_colors:
                            mapping[right_color] = left_color
                    else:
                        # pair: [panel_color][new_color]
                        if left_color in extracted_colors:
                            mapping[left_color] = right_color

            if not mapping:
                return None

            output = main_extract.copy()
            for old_color, new_color in mapping.items():
                output[output == old_color] = new_color
            return output

        # --- learn from training: which panel rank and which direction ---
        learned_rank      = None
        learned_direction = None  # True = panel_on_right, False = panel_on_left

        for my_ts in arc_problem.training_set():
            train_in  = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            solid_panels = find_solid_panels(train_in)
            if not solid_panels:
                return None

            matched_rank      = None
            matched_direction = None

            for rank, (top_row, bottom_row, left_col, right_col) in enumerate(solid_panels):
                for panel_is_on_right in [True, False]:
                    result = build_mapping_and_output(
                        train_in, top_row, bottom_row, left_col, right_col, panel_is_on_right)
                    if result is None:
                        continue
                    if result.shape == train_out.shape and np.array_equal(result, train_out):
                        matched_rank      = rank
                        matched_direction = panel_is_on_right
                        break
                if matched_rank is not None:
                    break

            if matched_rank is None:
                return None

            if learned_rank is None:
                learned_rank      = matched_rank
                learned_direction = matched_direction
            elif learned_rank != matched_rank or learned_direction != matched_direction:
                return None

        if learned_rank is None:
            return None

        # --- apply to test grid ---
        solid_panels = find_solid_panels(grid)
        if learned_rank >= len(solid_panels):
            return None

        top_row, bottom_row, left_col, right_col = solid_panels[learned_rank]
        return build_mapping_and_output(grid, top_row, bottom_row, left_col, right_col, learned_direction)

    
    
    # reviewed
    def diagonal_and_straight_from_initial_to_final(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        992798f6

        """

        # I first use the training examples to learn the 3 important colors.
        all_my_training_sets = arc_problem.training_set()

        my_start_color = None
        my_end_color = None
        my_path_color = None

        for my_ts in all_my_training_sets:
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            input_colors = sorted(int(value) for value in np.unique(train_in) if value != 0)
            output_colors = sorted(int(value) for value in np.unique(train_out) if value != 0)

            # Input should have 2 colors and output should have those same 2 plus 1 path color.
            if len(input_colors) != 2:
                continue
            if len(output_colors) != 3:
                continue

            extra_output_colors = [color for color in output_colors if color not in input_colors]
            if len(extra_output_colors) != 1:
                continue

            my_path_color = extra_output_colors[0]

            # I keep the two input colors as my start and end colors.
            # For this puzzle, I just learn them dynamically instead of hardcoding.
            color_1, color_2 = input_colors

            color_1_points = np.argwhere(train_in == color_1)
            color_2_points = np.argwhere(train_in == color_2)

            if len(color_1_points) != 1 or len(color_2_points) != 1:
                continue

            my_start_color = color_2
            my_end_color = color_1
            break

        if my_start_color is None or my_end_color is None or my_path_color is None:
            return None

        # Now I find the start and end points in the test grid using the learned colors.
        start_points = np.argwhere(grid == my_start_color)
        end_points = np.argwhere(grid == my_end_color)

        if len(start_points) != 1 or len(end_points) != 1:
            return None

        start_row, start_col = map(int, start_points[0])
        end_row, end_col = map(int, end_points[0])

        # I measure how far apart the two points are.
        row_diff = end_row - start_row
        col_diff = end_col - start_col

        # This tells me whether I should move up/down and left/right.
        row_step = 0 if row_diff == 0 else (1 if row_diff > 0 else -1)
        col_step = 0 if col_diff == 0 else (1 if col_diff > 0 else -1)

        abs_row_diff = abs(row_diff)
        abs_col_diff = abs(col_diff)

        # This tells me how much can move diagonally,
        # how much is left as straight movement,
        # and whether the straight part should go vertical or horizontal.
        diagonal_line = min(abs_row_diff, abs_col_diff)
        straight_line = abs(abs_row_diff - abs_col_diff)
        move_towards_row = abs_row_diff >= abs_col_diff

        output = np.zeros_like(grid)
        output[start_row, start_col] = my_start_color
        output[end_row, end_col] = my_end_color

        current_row, current_col = start_row, start_col

        # First I take one diagonal step if that is possible.
        if diagonal_line > 0:
            current_row += row_step
            current_col += col_step
            if (current_row, current_col) != (end_row, end_col):
                output[current_row, current_col] = my_path_color
            diagonal_line -= 1

        # Then I use the extra straight steps in the longer direction.
        for _ in range(straight_line):
            if move_towards_row:
                current_row += row_step
            else:
                current_col += col_step

            if (current_row, current_col) != (end_row, end_col):
                output[current_row, current_col] = my_path_color

        # Finally I finish with the remaining diagonal steps.
        for _ in range(diagonal_line):
            current_row += row_step
            current_col += col_step

            if (current_row, current_col) != (end_row, end_col):
                output[current_row, current_col] = my_path_color

        return output
    


    def count_markers_inside_one_box(self, grid: np.ndarray, arc_problem: ArcProblem) -> np.ndarray | None:
        """
        c8b7cc0f

        """

        # I first check my rule on the training sets.
        all_my_training_sets = arc_problem.training_set()

        for my_ts in all_my_training_sets:
            train_in = my_ts.get_input_data().data()
            train_out = my_ts.get_output_data().data()

            ones = np.argwhere(train_in == 1)
            if len(ones) == 0:
                return None

            marker_colors = [int(color) for color in np.unique(train_in) if int(color) not in (0, 1)]
            if len(marker_colors) != 1:
                return None
            marker = marker_colors[0]

            top_row, left_col = ones.min(axis=0)
            bottom_row, right_col = ones.max(axis=0)
            top_row, left_col, bottom_row, right_col = map(int, (top_row, left_col, bottom_row, right_col))

            # Count how many marker cells are inside the full box area.
            inside_count = int(np.sum(train_in[top_row:bottom_row + 1, left_col:right_col + 1] == marker))
            if inside_count <= 0:
                return None

            output = np.zeros((3, 3), dtype=int)

            # Fill the 3x3 output row by row using the marker color.
            fill_count = min(inside_count, 9)
            index = 0
            for row in range(3):
                for col in range(3):
                    if index < fill_count:
                        output[row, col] = marker
                    index += 1

            # My training rule should match the training output.
            if output.shape != train_out.shape:
                return None
            if not np.array_equal(output, train_out):
                return None

        # If the rule works on training, now I apply the same logic to the test grid.
        ones = np.argwhere(grid == 1)
        if len(ones) == 0:
            return None

        marker_colors = [int(color) for color in np.unique(grid) if int(color) not in (0, 1)]
        if len(marker_colors) != 1:
            return None
        marker = marker_colors[0]

        top_row, left_col = ones.min(axis=0)
        bottom_row, right_col = ones.max(axis=0)
        top_row, left_col, bottom_row, right_col = map(int, (top_row, left_col, bottom_row, right_col))

        # Count how many marker cells are inside the full box area.
        inside_count = int(np.sum(grid[top_row:bottom_row + 1, left_col:right_col + 1] == marker))
        if inside_count <= 0:
            return None

        output = np.zeros((3, 3), dtype=int)

        # Fill the 3x3 output row by row using the marker color.
        fill_count = min(inside_count, 9)
        index = 0
        for row in range(3):
            for col in range(3):
                if index < fill_count:
                    output[row, col] = marker
                index += 1

        return output
