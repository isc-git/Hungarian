use nalgebra::RawStorageMut;

#[derive(Debug, Clone)]
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
        for v in r.iter_mut() {
            *v -= min;
        }
    });

    // subtract minimum value from each respective col
    costs.column_iter_mut().for_each(|mut c| {
        let min = c.min();
        for v in c.iter_mut() {
            *v -= min;
        }
    });

    // try to assign abritrary zeroes on distinct rows and columns
    let (h, w) = costs.shape();
    let mut starred = Vec::new();
    for r in 0..h {
        for c in 0..w {
            if starred.iter().any(|(_, star_col, _)| *star_col == c) {
                continue;
            }

            if costs.get((r, c)).expect("within bounds").abs() < T::default_epsilon() {
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
                if costs.get((r, c)).expect("within bounds").abs() < T::default_epsilon() {
                    // found an uncovered zero
                    primed.push((r, c));
                    match starred.iter_mut().find(|(s_r, _s_c, _s_d)| *s_r == r) {
                        Some(star) => {
                            star.2 = Direction::Horizontal;
                            continue 'outer;
                        }
                        None => {
                            let mut path = vec![(r, c)];
                            let mut current = (r, c);
                            while let Some(star) =
                                starred.iter().find(|(_s_r, s_c, _s_d)| *s_c == current.1)
                            {
                                current.0 = star.0;
                                path.push(current);

                                let prime = primed
                                    .iter()
                                    .find(|(p_r, _p_c)| *p_r == current.0)
                                    .expect("known");
                                current.1 = prime.1;
                                path.push(current);
                            }

                            for (path_r, path_c) in path.into_iter() {
                                if let Some(index) = starred
                                    .iter()
                                    .position(|(s_r, s_c, _)| *s_r == path_r && *s_c == path_c)
                                {
                                    starred.remove(index);
                                }

                                if primed
                                    .iter()
                                    .any(|(p_r, p_c)| *p_r == path_r && *p_c == path_c)
                                {
                                    starred.push((path_r, path_c, Direction::Vertical));
                                }
                            }

                            primed.clear();
                            starred
                                .iter_mut()
                                .for_each(|(_, _, d)| *d = Direction::Vertical);

                            continue 'outer;
                        }
                    }
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

                min = min.min(*costs.get((r, c)).expect("within bounds"));
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
                    (true, true) => *costs.get_mut((r, c)).expect("within bounds") += min,
                    (true, false) | (false, true) => {}
                    (false, false) => *costs.get_mut((r, c)).expect("within bounds") -= min,
                }
            }
        }
    }

    // sort by columns
    starred.sort_by_key(|a| a.1);
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
