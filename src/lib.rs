use nalgebra::RawStorageMut;

#[derive(Debug, Clone, PartialEq)]
enum Direction {
    Vertical,
    Horizontal,
}

pub fn hungarian<T, D, S>(costs: &mut nalgebra::SquareMatrix<T, D, S>) -> Vec<usize>
where
    T: nalgebra::RealField + std::ops::Sub<T, Output = T> + Copy,
    D: nalgebra::Dim,
    S: nalgebra::RawStorage<T, D, D> + RawStorageMut<T, D, D>,
{
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
    let (h, w) = costs.shape();
    let mut starred = Vec::new();
    for r in 0..h {
        for c in 0..w {
            if starred.iter().any(|(_, star_col, _)| *star_col == c) {
                continue;
            }

            if costs[(r, c)].abs() < T::default_epsilon() {
                starred.push((r, c, Direction::Vertical));
                // breaks such that no more values are checked on this row
                break;
            }
        }
    }

    // find all non-starred zeros and prime them
    let mut primed = Vec::new();
    'outer: loop {
        for r in 0..h {
            // if a prime exists in this row, it is covered
            if primed.iter().any(|(p_row, _p_col)| *p_row == r) {
                continue;
            }

            for c in 0..w {
                // if a star not covered by a prime exists on this column, skip
                if starred.iter().any(|(s_row, s_col, s_dir)| match s_dir {
                    Direction::Vertical => *s_col == c,
                    Direction::Horizontal => *s_col == c && *s_row == r,
                }) {
                    continue;
                }

                // check if this value is zero
                if costs[(r, c)].abs() < T::default_epsilon() {
                    // found an uncovered zero
                    primed.push((r, c));
                    match starred.iter_mut().find(|(s_r, _s_c, _s_d)| *s_r == r) {
                        Some(star) => {
                            star.2 = Direction::Horizontal;
                        }
                        None => {
                            let mut current = (r, c);
                            starred.push((r, c, Direction::Vertical));
                            // a vertical star is an unpathed star
                            starred
                                .iter_mut()
                                .for_each(|(_, _, d)| *d = Direction::Vertical);

                            while let Some(star_index) =
                                starred.iter().position(|(_s_r, s_c, s_d)| {
                                    *s_c == current.1 && *s_d == Direction::Vertical
                                })
                            {
                                let star = starred.remove(star_index);
                                current.0 = star.0;

                                let prime_index = primed
                                    .iter()
                                    .position(|(p_r, _p_c)| *p_r == current.0)
                                    .expect("known");

                                let prime = primed.remove(prime_index);
                                current.1 = prime.1;
                                starred.push((current.0, current.1, Direction::Horizontal));
                            }

                            primed.clear();
                            starred
                                .iter_mut()
                                .for_each(|(_, _, d)| *d = Direction::Vertical);
                        }
                    }
                    continue 'outer;
                }
            }
        }

        if starred.len() == h {
            break;
        }

        let mut min = T::max_value().expect("real value has maximum");
        for r in 0..h {
            if primed.iter().any(|(p_r, _p_c)| *p_r == r) {
                continue;
            }

            for c in 0..w {
                if starred.iter().any(|(b, a, d)| match d {
                    Direction::Vertical => *a == c,
                    Direction::Horizontal => *b == r && *a == c,
                }) {
                    continue;
                }

                min = min.min(costs[(r, c)]);
            }
        }

        for r in 0..h {
            for c in 0..w {
                let covered_row = primed.iter().any(|(a, _)| *a == r)
                    || starred.iter().any(|(a, b, d)| match d {
                        Direction::Vertical => *b == c,
                        Direction::Horizontal => *a == r,
                    });

                let covered_col = starred.iter().any(|(b, a, d)| match d {
                    Direction::Vertical => *a == c,
                    Direction::Horizontal => *a == c && *b == r,
                });

                match (covered_row, covered_col) {
                    (true, true) => costs[(r, c)] += min,
                    (true, false) | (false, true) => {}
                    (false, false) => costs[(r, c)] -= min,
                }
            }
        }
    }

    // sort by columns
    starred.sort_unstable_by_key(|a| a.1);
    starred.into_iter().map(|(r, _, _)| r).collect::<Vec<_>>()
    //starred.into_iter().map(|(r, c, _)| c).collect::<Vec<_>>()
}

#[cfg(test)]
mod test {
    use nalgebra::{Dim, Matrix, Matrix2, Matrix4, Matrix5, RawStorage};

    use super::*;

    fn assert_costs<R, C, S>(
        costs: &Matrix<f64, R, C, S>,
        assignments: &[usize],
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
            .enumerate()
            .map(|(i, a)| costs.get((*a, i)).expect("within cost bounds"))
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
        let assignments = hungarian(&mut costs.clone());
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
        let assignments = hungarian(&mut costs.clone());
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
        let assignments = hungarian(&mut costs.clone());
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
        let assignments = hungarian(&mut costs.clone());
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
        let assignments = hungarian(&mut costs.clone());
        let expected_cost = 86.;
        assert!(assert_costs(
            &costs,
            &assignments,
            expected_cost,
            f64::EPSILON
        ));
    }
}
