use nalgebra::RawStorageMut;

#[derive(Debug, Clone, Default)]
pub struct Allocations<T> {
    row: Vec<usize>,
    col: Vec<usize>,
    row_prime: Vec<usize>,
    col_prime: Vec<usize>,
    covered_rows: Vec<bool>,
    covered_cols: Vec<bool>,
    rows_offsets: Vec<T>,
    cols_offsets: Vec<T>,
}

impl<T> Allocations<T> {
    pub fn assignment(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.row.iter().cloned().zip(self.col.iter().cloned())
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.row.clear();
        self.col.clear();
        self.row_prime.clear();
        self.col_prime.clear();
        self.covered_rows.clear();
        self.covered_cols.clear();
    }

    #[inline(always)]
    pub fn submit_star(&mut self, row: usize, col: usize) {
        self.row.push(row);
        self.col.push(col);
    }

    #[inline(always)]
    pub fn submit_prime(&mut self, row: usize, col: usize) {
        self.row_prime.push(row);
        self.col_prime.push(col);
    }

    #[inline(always)]
    fn remove_star(&mut self, i: usize) -> (usize, usize) {
        let row = self.row.remove(i);
        let col = self.col.remove(i);
        (row, col)
    }

    #[inline(always)]
    fn remove_prime(&mut self, i: usize) -> (usize, usize) {
        let row = self.row_prime.remove(i);
        let col = self.col_prime.remove(i);
        (row, col)
    }
}

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
    let (h, w) = costs.shape();
    assignments.clear();
    assignments.covered_rows.resize(h, false);
    assignments.covered_cols.resize(w, false);

    assignments.rows_offsets.clear();
    assignments.rows_offsets.resize(h, T::zero());
    assignments.cols_offsets.clear();
    assignments.cols_offsets.resize(w, T::zero());

    // subtract minimum value from each respective row
    costs.row_iter().enumerate().for_each(|(i, r)| {
        let min = r.min();
        assignments.rows_offsets[i] = min;
    });

    // subtract minimum value from each respective col
    costs.column_iter().enumerate().for_each(|(i, c)| {
        let mut min = T::max_value();
        for r in 0..h {
            min = min.simd_min(c[r] - assignments.rows_offsets[r]);
        }

        assignments.cols_offsets[i] = min;
    });

    // try to assign abritrary zeroes on distinct rows and columns
    for col in 0..w {
        for row in 0..h {
            if assignments.row.contains(&row) {
                continue;
            }

            if (costs[(row, col)] - assignments.rows_offsets[row] - assignments.cols_offsets[col])
                .is_zero()
            {
                assignments.covered_cols[col] = true;
                assignments.submit_star(row, col);
                // breaks such that no more values are checked on this column
                break;
            }
        }
    }

    // find all non-starred zeros and prime them
    loop {
        let mut uncovered_zero = None;
        'zero_finder: for col in 0..w {
            if assignments.covered_cols[col] {
                continue;
            }

            for row in 0..h {
                if assignments.covered_rows[row] {
                    continue;
                }

                // check if this value is zero
                if (costs[(row, col)]
                    - assignments.rows_offsets[row]
                    - assignments.cols_offsets[col])
                    .is_zero()
                {
                    uncovered_zero = Some((row, col));
                    break 'zero_finder;
                }
            }
        }

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

                        let (prime_index, _) = assignments
                            .row_prime
                            .iter()
                            .enumerate()
                            .find(|(_, row)| **row == current.0)
                            .expect("known");

                        let (_, prime_col) = assignments.remove_prime(prime_index);
                        current.1 = prime_col;
                        to_add = current
                    }

                    assignments.submit_star(to_add.0, to_add.1);

                    assignments.covered_rows.fill(false);
                    assignments.covered_cols.fill(false);
                    assignments.row_prime.clear();
                    assignments.col_prime.clear();
                    for assigned_col in assignments.col.iter() {
                        assignments.covered_cols[*assigned_col] = true;
                    }
                }
            }

            continue;
        }

        if assignments.row.len() == h {
            break;
        }

        let mut min = <T as num_traits::Bounded>::max_value();
        for col in 0..w {
            if assignments.covered_cols[col] {
                continue;
            }

            for row in 0..h {
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
        for i in 0..h {
            if !assignments.covered_rows[i] {
                assignments.rows_offsets[i] += min;
            }
        }

        // add min to all covered columns
        for i in 0..w {
            if assignments.covered_cols[i] {
                assignments.cols_offsets[i] -= min;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use nalgebra::{Dim, Matrix, Matrix2, Matrix3, Matrix4, Matrix5, RawStorage};

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
        hungarian(&mut costs.clone(), &mut assignments);
        assert!(assignments.assignment().collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn unary() {
        #[rustfmt::skip]
        let costs = nalgebra::DMatrix::<f64>::from_row_slice(1, 1, &[1.]);
        let mut assignments = Allocations::default();
        hungarian(&mut costs.clone(), &mut assignments);
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
        hungarian(&mut costs.clone(), &mut assignments);
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
        hungarian(&mut costs.clone(), &mut assignments);
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
        hungarian(&mut costs.clone(), &mut assignments);
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
        hungarian(&mut costs.clone(), &mut assignments);
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
        hungarian(&mut costs.clone(), &mut assignments);
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
        hungarian(&mut costs.clone(), &mut assignments);
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
        hungarian(&mut costs.clone(), &mut assignments);
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
        hungarian(&mut costs.clone(), &mut assignments);
        let expected_cost = 78.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
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
        hungarian(&mut costs.clone(), &mut assignments);
        let expected_cost = 137.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }
}
