use nalgebra::RawStorageMut;

#[derive(Debug, Clone, PartialEq)]
enum AllocationStatus {
    Star,
    Prime,
}

#[derive(Debug, Clone)]
pub struct Allocation {
    row: usize,
    col: usize,
    status: AllocationStatus,
}

impl Allocation {
    pub fn assignment(&self) -> (usize, usize) {
        (self.row, self.col)
    }
}

pub fn hungarian<T, D, S>(
    costs: &mut nalgebra::SquareMatrix<T, D, S>,
    assignments: &mut Vec<Allocation>,
) where
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
    let mut covered_rows = vec![false; h];
    let mut covered_cols = vec![false; w];

    // subtract minimum value from each respective row
    costs.row_iter_mut().for_each(|mut r| {
        let min = r.min();
        r.add_scalar_mut(-min)
    });

    // subtract minimum value from each respective col
    costs.column_iter_mut().for_each(|mut c| {
        let min = c.min();
        c.add_scalar_mut(-min);
    });

    // try to assign abritrary zeroes on distinct rows and columns
    for col in 0..w {
        for row in 0..h {
            if assignments.iter().any(|a: &Allocation| a.row == row) {
                continue;
            }

            if costs[(row, col)].is_zero() {
                covered_cols[col] = true;
                assignments.push(Allocation {
                    row,
                    col,
                    status: AllocationStatus::Star,
                });
                // breaks such that no more values are checked on this column
                break;
            }
        }
    }

    // find all non-starred zeros and prime them
    loop {
        let mut uncovered_zero = None;
        'zero_finder: for col in 0..w {
            if covered_cols[col] {
                continue;
            }

            for row in 0..h {
                if covered_rows[row] {
                    continue;
                }

                // check if this value is zero
                if costs[(row, col)].is_zero() {
                    uncovered_zero = Some((row, col));
                    break 'zero_finder;
                }
            }
        }

        if let Some((row, col)) = uncovered_zero {
            match assignments.iter().find(|a| match a.status {
                AllocationStatus::Star => a.row == row,
                AllocationStatus::Prime => false,
            }) {
                Some(star) => {
                    covered_cols[star.col] = false;
                    covered_rows[star.row] = true;
                    assignments.push(Allocation {
                        row,
                        col,
                        status: AllocationStatus::Prime,
                    });
                }
                None => {
                    let mut current = (row, col);
                    // add the starting position as a new star
                    assignments.push(Allocation {
                        row,
                        col,
                        status: AllocationStatus::Star,
                    });

                    while let Some(star_index) = assignments.iter().position(|a| {
                        a.status == AllocationStatus::Star
                            && a.col == current.1
                            && a.row != current.0
                    }) {
                        let star = assignments.remove(star_index);
                        current.0 = star.row;

                        let prime = assignments
                            .iter_mut()
                            .find(|a| a.status == AllocationStatus::Prime && a.row == current.0)
                            .expect("known");

                        prime.status = AllocationStatus::Star;
                        current.1 = prime.col;
                    }

                    assignments.retain(|a| a.status != AllocationStatus::Prime);
                    covered_rows.fill(false);
                    covered_cols.fill(false);
                    assignments.iter().for_each(|a| {
                        if a.status == AllocationStatus::Star {
                            covered_cols[a.col] = true;
                        }
                    });
                }
            }

            continue;
        }

        if assignments
            .iter()
            .filter(|d| match d.status {
                AllocationStatus::Star => true,
                AllocationStatus::Prime => false,
            })
            .count()
            == h
        {
            break;
        }

        let mut min = <T as num_traits::Bounded>::max_value();
        for col in 0..w {
            if covered_cols[col] {
                continue;
            }

            for row in 0..h {
                if covered_rows[row] {
                    continue;
                }

                let curr = costs[(row, col)];
                if curr < min {
                    min = curr;
                }
            }
        }

        // subtract min from all uncovered rows
        costs.row_iter_mut().enumerate().for_each(|(i, mut r)| {
            if !covered_rows[i] {
                r.add_scalar_mut(-min)
            }
        });

        // add min to all covered columns
        costs.column_iter_mut().enumerate().for_each(|(i, mut c)| {
            if covered_cols[i] {
                c.add_scalar_mut(min);
            }
        });
    }

    assignments.retain(|a| match a.status {
        AllocationStatus::Star => true,
        AllocationStatus::Prime => false,
    });
}

#[cfg(test)]
mod test {
    use nalgebra::{Dim, Matrix, Matrix2, Matrix4, Matrix5, RawStorage};

    use super::*;

    fn assert_costs<R, C, S>(
        costs: &Matrix<f64, R, C, S>,
        assignments: &[Allocation],
        cost_expected: f64,
        epsilon: f64,
    ) -> bool
    where
        R: Dim,
        C: Dim,
        S: RawStorage<f64, R, C>,
    {
        (assignments
            .iter()
            .map(|a| costs.get(a.assignment()).expect("within cost bounds"))
            .sum::<f64>()
            - cost_expected)
            .abs()
            < epsilon
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
        let mut assignments = Vec::with_capacity(costs.shape().1);
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
        let mut assignments = Vec::with_capacity(costs.shape().1);
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
        let mut assignments = Vec::with_capacity(costs.shape().1);
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
        let mut assignments = Vec::with_capacity(costs.shape().1);
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
        let mut assignments = Vec::with_capacity(costs.shape().1);
        hungarian(&mut costs.clone(), &mut assignments);
        let expected_cost = 86.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }
}
