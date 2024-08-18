//! An implementation of the matrix intepretation of the Hungarian algorithm
//! ```
//! use hungarian::{Allocations, hungarian};
//! use nalgebra::DMatrix;
//!
//! let mut assignments = Allocations::default();
//! let costs = DMatrix::<i32>::from_row_slice(
//!     3, 3,
//!     &[
//!         8, 4, 7,
//!         5, 2, 3,
//!         9, 4, 8,
//!     ]
//! );
//!
//! hungarian(&costs, &mut assignments);
//! let cost = assignments.assignment().map(|i| costs[i]).sum::<i32>();
//! # assert_eq!(cost, 15);
//! ```
//!

use nalgebra::RawStorageMut;

#[derive(Debug, Clone, Default)]
/// retained buffers for repeated use by the algorithm
pub struct Allocations<T: num_traits::Zero + Copy> {
    row: Vec<usize>,
    col: Vec<usize>,
    prime: Vec<usize>,
    covered_rows: Vec<bool>,
    covered_cols: Vec<bool>,
    rows_offsets: Vec<T>,
    cols_offsets: Vec<T>,
}

impl<T: num_traits::Zero + Copy> Allocations<T> {
    /// returns an iterator of the assignments (row, column)
    pub fn assignment(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.row.iter().cloned().zip(self.col.iter().cloned())
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.row.clear();
        self.col.clear();
        self.covered_rows.clear();
        self.covered_cols.clear();
    }

    #[inline(always)]
    fn resize(&mut self, size: usize) {
        self.covered_rows.resize(size, false);
        self.covered_cols.resize(size, false);
        self.prime.resize(size, 0);
        self.rows_offsets.resize(size, T::zero());
        self.cols_offsets.resize(size, T::zero());
    }

    #[inline(always)]
    fn submit_star(&mut self, row: usize, col: usize) {
        self.row.push(row);
        self.col.push(col);
    }

    #[inline(always)]
    fn submit_prime(&mut self, row: usize, col: usize) {
        self.prime[row] = col;
    }

    #[inline(always)]
    fn remove_star(&mut self, i: usize) -> (usize, usize) {
        let row = self.row.remove(i);
        let col = self.col.remove(i);
        (row, col)
    }
}

/// minimizes the cost of assigning N workers to N jobs
///
/// # Arguments
/// - `costs`: cost matrix containing weights
/// - `assignments`: stores the assignments, recovered with [Allocations::assignment]
pub fn hungarian<T, D, S>(costs: &nalgebra::SquareMatrix<T, D, S>, assignments: &mut Allocations<T>)
where
    T: std::ops::Sub<T, Output = T>
        + Copy
        + nalgebra::SimdValue<Element = T>
        + nalgebra::SimdPartialOrd
        + num_traits::bounds::Bounded
        + num_traits::Zero
        + std::ops::SubAssign
        + std::ops::AddAssign
        + std::fmt::Debug
        + std::ops::Neg<Output = T>
        + PartialEq
        + PartialOrd
        + 'static,
    D: nalgebra::Dim,
    S: nalgebra::RawStorage<T, D, D> + RawStorageMut<T, D, D>,
{
    let (size, _) = costs.shape();
    assignments.clear();
    assignments.resize(size);

    // subtract minimum value from each respective row
    costs.row_iter().enumerate().for_each(|(i, r)| {
        let min = r.min();
        assignments.rows_offsets[i] = min;
    });

    // subtract minimum value from each respective col
    costs.column_iter().enumerate().for_each(|(i, c)| {
        let mut min = T::max_value();
        for r in 0..size {
            min = min.simd_min(c[r] - assignments.rows_offsets[r]);
        }

        assignments.cols_offsets[i] = min;
    });

    // try to assign abritrary zeroes on distinct rows and columns
    for row in 0..size {
        for col in 0..size {
            if assignments.covered_cols[col] {
                continue;
            }

            if costs[(row, col)] <= assignments.rows_offsets[row] + assignments.cols_offsets[col] {
                assignments.covered_cols[col] = true;
                assignments.submit_star(row, col);
                break;
            }
        }
    }

    loop {
        // find an uncovered zero
        let mut uncovered_zero = None;
        'zero_finder: for col in 0..size {
            if assignments.covered_cols[col] {
                continue;
            }

            for row in 0..size {
                if assignments.covered_rows[row] {
                    continue;
                }

                if costs[(row, col)]
                    <= assignments.rows_offsets[row] + assignments.cols_offsets[col]
                {
                    uncovered_zero = Some((row, col));
                    break 'zero_finder;
                }
            }
        }

        // action the new zero
        if let Some((row, col)) = uncovered_zero {
            match assignments.row.iter().enumerate().find(|(_, r)| row == **r) {
                Some((star_index, star_row)) => {
                    assignments.covered_cols[assignments.col[star_index]] = false;
                    assignments.covered_rows[*star_row] = true;
                    assignments.submit_prime(row, col)
                }
                None => {
                    let mut current = (row, col);
                    let mut to_add = current;
                    while let Some(star_index) =
                        assignments.col.iter().position(|col| *col == current.1)
                    {
                        let (star_row, _) = assignments.remove_star(star_index);
                        current.0 = star_row;
                        assignments.submit_star(to_add.0, to_add.1);

                        let prime_col = assignments.prime[current.0];
                        current.1 = prime_col;
                        to_add = current
                    }

                    assignments.submit_star(to_add.0, to_add.1);

                    assignments.covered_rows.fill(false);
                    assignments.covered_cols.fill(false);
                    for assigned_col in assignments.col.iter() {
                        assignments.covered_cols[*assigned_col] = true;
                    }
                }
            }

            continue;
        }

        if assignments.row.len() == size {
            break;
        }

        // determine the minimum of non-covered elements
        let mut min = <T as num_traits::Bounded>::max_value();
        for col in 0..size {
            if assignments.covered_cols[col] {
                continue;
            }

            for row in 0..size {
                if assignments.covered_rows[row] {
                    continue;
                }

                min = min.simd_min(
                    costs[(row, col)]
                        - assignments.rows_offsets[row]
                        - assignments.cols_offsets[col],
                );
            }
        }

        // subtract min from all uncovered rows
        for i in 0..size {
            if !assignments.covered_rows[i] {
                assignments.rows_offsets[i] += min;
            }
        }

        // add min to all covered columns
        for i in 0..size {
            if assignments.covered_cols[i] {
                assignments.cols_offsets[i] -= min;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use nalgebra::{DMatrix, Dim, Matrix, Matrix2, Matrix3, Matrix4, Matrix5, RawStorage};

    use super::*;

    fn assert_costs<R, C, S>(
        costs: &Matrix<f64, R, C, S>,
        assignments: &Allocations<f64>,
        cost_expected: f64,
        epsilon: f64,
    ) -> bool
    where
        R: Dim,
        C: Dim,
        S: RawStorage<f64, R, C>,
    {
        (assignments
            .assignment()
            .map(|a| costs.get(a).expect("within cost bounds"))
            .sum::<f64>()
            - cost_expected)
            .abs()
            < epsilon
    }

    #[test]
    fn null() {
        #[rustfmt::skip]
        let costs = nalgebra::DMatrix::<f64>::from_row_slice(0, 0, &[]);
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        assert!(assignments.assignment().collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn unary() {
        #[rustfmt::skip]
        let costs = nalgebra::DMatrix::<f64>::from_row_slice(1, 1, &[1.]);
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let assignments = assignments.assignment().collect::<Vec<_>>();
        assert!(assignments.len() == 1);
        assert!(assignments[0] == (0, 0));
    }

    #[test]
    fn basic_two() {
        #[rustfmt::skip]
        let costs = Matrix2::from_row_slice(
            &[
                1., 2.,
                2., 1.,
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 2.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }

    #[test]
    fn basic_two_rev() {
        #[rustfmt::skip]
        let costs = Matrix2::from_row_slice(
            &[
                1., 2.,
                2., 100.
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 4.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }

    #[test]
    fn basic_four() {
        #[rustfmt::skip]
        let costs = Matrix4::from_row_slice(
            &[
                82., 83., 69., 92.,
                77., 37., 49., 92.,
                11., 69.,  5., 86.,
                 8.,  9., 98., 23.,
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 140.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }

    #[test]
    fn wikipedia_three() {
        #[rustfmt::skip]
        let costs = Matrix3::from_row_slice(
            &[
                8., 5., 9.,
                4., 2., 4.,
                7., 3., 8., 
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 15.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }

    #[test]
    fn basic_five() {
        #[rustfmt::skip]
        let costs = Matrix5::from_row_slice(
            &[
                10., 5.,13.,15.,16.,
                 3., 9.,18.,13., 6.,
                10., 7., 2., 2., 2.,
                 7.,11., 9., 7.,12.,
                 7., 9.,10., 4.,12.,
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 23.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }

    #[test]
    fn another_three() {
        #[rustfmt::skip]
        let costs = Matrix3::from_row_slice(
            &[
                10., 6., 9.,
                 5., 7., 8.,
                 9., 8., 5.,
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 16.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }

    #[test]
    fn basic_five_2() {
        #[rustfmt::skip]
        let costs = Matrix5::from_row_slice(
            &[
                20., 15., 18., 20., 25.,
                18., 20., 12., 14., 15.,
                21., 23., 25., 27., 25.,
                17., 18., 21., 23., 20.,
                18., 18., 16., 19., 20.,
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 86.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }

    #[test]
    fn another_four() {
        #[rustfmt::skip]
        let costs = Matrix4::from_row_slice(
            &[
                20., 15., 19., 25.,
                25., 18., 17., 23.,
                22., 23., 21., 24.,
                28., 17., 24., 24.,
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 78.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }

    #[test]
    fn another_four_int() {
        #[rustfmt::skip]
        let costs = Matrix4::from_row_slice(
            &[
                20, 15, 19, 25,
                25, 18, 17, 23,
                22, 23, 21, 24,
                28, 17, 24, 24,
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 78;
        assert_eq!(
            assignments
                .assignment()
                .map(|a| costs.get(a).expect("within cost bounds"))
                .sum::<i32>(),
            expected_cost
        );
    }

    #[test]
    fn y12() {
        #[rustfmt::skip]
        let costs = Matrix4::from_row_slice(
            &[
                12., 14., 74., 68.,
                10., 79., 73., 75.,
                92.,  9., 61., 34.,
                28., 84., 79., 81.,
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 137.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }

    #[test]
    fn y12_int() {
        #[rustfmt::skip]
        let costs = Matrix4::from_row_slice(
            &[
                12, 14, 74, 68,
                10, 79, 73, 75,
                92,  9, 61, 34,
                28, 84, 79, 81,
            ]
        );
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 137;
        assert_eq!(
            assignments
                .assignment()
                .map(|a| costs.get(a).expect("within cost bounds"))
                .sum::<i32>(),
            expected_cost
        );
    }

    // adapted from https://github.com/nwtnni/hungarian/tree/master to ensure valid
    // benchmarking comparison
    #[test]
    fn test_worst_case() {
        let mut costs = DMatrix::zeros(4, 4);
        let (rows, columns) = costs.shape();
        for col in 0..columns {
            for row in 0..rows {
                costs[(row, col)] = ((row + 1) * (col + 1)) as i32;
            }
        }
        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 20;
        assert_eq!(
            assignments
                .assignment()
                .map(|a| costs.get(a).expect("within cost bounds"))
                .sum::<i32>(),
            expected_cost
        );
    }
    // found https://github.com/nwtnni/hungarian/blob/master/src/lib.rs
    // originally from https://stackoverflow.com/questions/26893961/cannot-solve-hungarian-algorithm
    #[test]
    fn cannot_solve() {
        #[rustfmt::skip]
        let costs = DMatrix::from_row_slice(14, 14,
            &[
                  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0,
                 53, 207, 256, 207, 231, 348, 348, 348, 231, 244, 244, 0, 0, 0,
                240,  33,  67,  33,  56, 133, 133, 133,  56,  33,  33, 0, 0, 0,
                460, 107, 200, 107, 122, 324, 324, 324, 122,  33,  33, 0, 0, 0,
                167, 340, 396, 340, 422, 567, 567, 567, 422, 442, 442, 0, 0, 0,
                167, 367, 307, 367, 433, 336, 336, 336, 433, 158, 158, 0, 0, 0,
                160,  20,  37,  20,  31,  70,  70,  70,  31,  22,  22, 0, 0, 0,
                200, 307, 393, 307, 222, 364, 364, 364, 222, 286, 286, 0, 0, 0,
                 33, 153, 152, 153, 228, 252, 252, 252, 228,  78,  78, 0, 0, 0,
                 93, 140, 185, 140,  58, 118, 118, 118,  58,  44,  44, 0, 0, 0,
                  0,   7,  22,   7,  19,  58,  58,  58,  19,   0,   0, 0, 0, 0,
                 67, 153, 241, 153, 128, 297, 297, 297, 128,  39,  39, 0, 0, 0,
                 73, 253, 389, 253, 253, 539, 539, 539, 253,  36,  36, 0, 0, 0,
                173, 267, 270, 267, 322, 352, 352, 352, 322, 231, 231, 0, 0, 0,
            ]
        );

        let mut assignments = Allocations::default();
        hungarian(&costs, &mut assignments);
        let expected_cost = 828;
        assert_eq!(
            assignments
                .assignment()
                .map(|a| costs.get(a).expect("within cost bounds"))
                .sum::<i32>(),
            expected_cost
        );
    }
}
