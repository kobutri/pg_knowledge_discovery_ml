use rand::{thread_rng, Rng, SeedableRng};
use rand_distr::Distribution;
use serde::Deserialize;
use std::slice::SliceIndex;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

#[derive(Deserialize)]
struct Doc {
    id: String,
    token: Vec<String>,
    author: String,
}

fn main() {
    let n_gibbs: usize = 2000;

    let train_input =
        std::fs::read_to_string("../../preprocessing_output/preprocessed_train_L.json").unwrap();
    let train_docs: Vec<Doc> = serde_json::from_str(&train_input).unwrap();

    let mut dict = HashSet::new();
    train_docs.iter().for_each(|doc| {
        doc.token.iter().for_each(|token| {
            dict.insert(token);
        })
    });
    let mut dict_map = HashMap::with_capacity(dict.len());
    let mut dict_arr = vec![];
    dict.iter().for_each(|token| {
        dict_map.insert(token.clone(), dict_arr.len());
        dict_arr.push(token.to_owned().to_owned());
    });
    let vocab = dict_arr;
    let docs =
        Vec::from_iter(train_docs.iter().map(|doc| {
            Vec::from_iter(doc.token.iter().map(|token| *dict_map.get(token).unwrap()))
        }));
    let V = vocab.len();
    let k = 100;
    let N = Vec::from_iter(docs.iter().map(|doc| doc.len()));
    let M = docs.len();

    println!("V: {}\nk: {}\nN: {:?},...\nM: {}", V, k, &N[0..10], M);

    let mut rng = rand_xorshift::XorShiftRng::from_rng(thread_rng()).unwrap();
    let alpha = rand_distr::Gamma::new(100.0, 0.1f64)
        .unwrap()
        .sample(&mut rng);
    let beta = rand_distr::Gamma::new(100.0, 0.01)
        .unwrap()
        .sample(&mut rng);
    println!("α: {}\nβ: {}", alpha, beta);

    let mut n_iw = vec![vec![0;k]; V];
    let mut n_i = vec![0usize; k];
    let mut n_di = vec![vec![0;k]; M];
    let mut n_d = vec![0usize; M];

    let N_max = *N.iter().max().unwrap();
    let mut assign = vec![0usize; M * N_max];

    for d in 0..M {
        for n in 0..N[d] {
            let w_dn = docs[d][n];
            assign[d * N_max + n] = rng.gen_range(0..k);
            let i = assign[d * N_max + n];
            n_iw[w_dn][i] += 1;
            n_i[i] += 1;
            n_di[d][i] += 1;
            n_d[d] += 1;
        }
    }

    println!("\n========== START SAMPLER ==========");
    for t in 0..n_gibbs {
        let current = t % 2;
        let next = (t + 1) % 2;
        for d in 0..M {
            for n in 0..N[d] {
                let w_dn = docs[d][n];

                let i_t = assign[d * N_max + n];
                n_iw[w_dn][i_t] -= 1;
                n_i[i_t] -= 1;
                n_di[d][i_t] -= 1;
                n_d[d] -= 1;

                let prob = {
                    let mut prob = vec![0f64; k];
                    for i in 0..k {
                        let left_num = n_iw[w_dn][i] as f64 + beta;
                        let left_denom = n_i[i] as f64 + V as f64 * beta;
                        let right_num = n_di[d][i] as f64 + alpha;
                        let right_denom = n_d[d] as f64 + k as f64 * alpha;
                        prob[i] = (left_num * right_num) / (left_denom * right_denom);
                    }
                    prob
                };
                let i_tp1 = rand::distributions::WeightedIndex::new(prob)
                    .unwrap()
                    .sample(&mut rng);

                n_iw[w_dn][i_tp1] += 1;
                n_i[i_tp1] += 1;
                n_di[d][i_tp1] += 1;
                n_d[d] += 1;
                assign[d * N_max + n] = i_tp1;
            }
        }
        if (t + 1) % 50 == 0 {
            println!("Sampled {}/{}", (t + 1), n_gibbs);
        }
    }

    let mut theta = vec![0f64;k*V];
    for d in 0..M {
        for i in 0..k {
            theta[d*k+i] = (n_di[d][i] as f64 + alpha) / (n_d[d] as f64 + k as f64 * alpha);
        }
    }
    let theta = serde_json::to_string(&theta).unwrap();
    std::fs::write("./theta_train", theta).unwrap();
}
