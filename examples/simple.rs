const ASSIGNMENT_SIZE: usize = 64;
const N: usize = 100;

fn main() {
    let mut assignments = Vec::new();
    let mut total_cost = 0.;
    for _ in 0..N {
        let costs = nalgebra::DMatrix::<f64>::new_random(ASSIGNMENT_SIZE, ASSIGNMENT_SIZE);
        hungarian::hungarian(&mut costs.clone(), &mut assignments);
        total_cost += assignments
            .iter()
            .map(|a| costs.get(a.assignment()).expect("within cost bounds"))
            .sum::<f64>();
    }

    println!("total: {total_cost}");
}
