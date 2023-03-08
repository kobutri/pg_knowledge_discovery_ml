use std::{
    collections::{HashMap, HashSet},
    ffi::OsString,
    vec,
};

use rand::{seq::SliceRandom, thread_rng, SeedableRng};
use rand_distr::{Distribution, Uniform, WeightedIndex};
use rand_xorshift::XorShiftRng;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
// use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::Deserialize;

fn main() {
    let mut preprocessing_map: HashMap<String, (OsString, OsString)> = HashMap::new();
    for path in std::fs::read_dir("../preprocessing_output").unwrap() {
        let path = path.unwrap();
        let str_path = path.file_name().clone().into_string().unwrap();
        let method = str_path
            .split("_")
            .last()
            .unwrap()
            .split(".")
            .next()
            .unwrap()
            .to_string();
        let entry = preprocessing_map
            .entry(method)
            .or_insert((OsString::new(), OsString::new()));
        if str_path.contains("train") {
            entry.0 = path.path().as_os_str().to_owned();
        } else {
            entry.1 = path.path().as_os_str().to_owned();
        }
    }
    std::fs::create_dir_all("results_output").unwrap();
    preprocessing_map
        .par_iter()
        .for_each(|(method, (train_path, test_path))| {
            let train_str = std::fs::read_to_string(train_path).unwrap();
            println!("{:?}", train_path);
            let train_docs: Vec<Document> = serde_json::from_str(&train_str).unwrap();

            let train_docs = Document::merge_docs(train_docs);
            let mut model = LDAModel::new(train_docs.clone());
            model.fit();
            let train_theta = model.get_theta();

            let test_str = std::fs::read_to_string(test_path).unwrap();
            let test_docs: Vec<Document> = serde_json::from_str(&test_str).unwrap();
            let mut predictions = Vec::with_capacity(test_docs.len());
            let mut rng = XorShiftRng::from_rng(thread_rng()).unwrap();
            for doc in test_docs {
                let test_theta = model.infer_theta(doc, &mut rng);
                let mut min = f64::INFINITY;
                let mut min_i = 0;
                for (i, theta) in train_theta.iter().enumerate() {
                    let dist = hellinger(theta, &test_theta);
                    if hellinger(theta, &test_theta) < min {
                        min = dist;
                        min_i = i;
                    }
                }
                predictions.push(train_docs[min_i].author.clone());
            }
            let predictions_string = serde_json::to_string(&predictions).unwrap();
            let output_path = format!("./results_output/results_lda_{}.json", method);
            std::fs::write(&output_path, predictions_string).unwrap();
        })
}

#[derive(Default, Clone, Debug)]
struct LDAModel {
    k: usize,
    v: usize,
    alpha: f64,
    beta: f64,
    vocab: Vec<String>,
    vocab_map: HashMap<String, usize>,
    docs: Vec<Vec<LDADoc>>,
    cdk: Vec<Vec<usize>>,
    ckv: Vec<Vec<usize>>,
    cd: Vec<usize>,
    ck: Vec<usize>,
}

impl LDAModel {
    fn new(docs: Vec<Document>) -> Self {

        let vocab =
            HashSet::<String>::from_iter(docs.iter().map(|doc| doc.token.clone()).flatten())
                .into_iter()
                .collect::<Vec<_>>();
        let vocab_map =
            vocab
                .iter()
                .enumerate()
                .fold(HashMap::new(), |mut map, (index, word)| {
                    map.insert(word.clone(), index);
                    map
                });

        let docs: Vec<Vec<_>> = docs
            .iter()
            .enumerate()
            .map(|(index, doc)| {
                doc.token
                    .iter()
                    .map(|token| LDADoc {
                        d: index,
                        w: vocab_map[token],
                        t: 0,
                    })
                    .collect()
            })
            .collect();

        let k = 50usize;
        let v = vocab.len();
        let alpha = 0.1;
        let beta = 0.01;

        let cdk = Vec::from_iter((0..docs.len()).map(|_| vec![0usize; k]));
        let ckv = Vec::from_iter((0..v).map(|_| vec![0usize; k]));
        let cd = vec![0usize; docs.len()];
        let ck = vec![0usize; k];

        Self {
            k,
            v,
            alpha,
            beta,
            vocab,
            vocab_map,
            docs,
            cdk,
            cd,
            ckv,
            ck,
        }
    }

    fn fit(&mut self) {
        let mut rng = XorShiftRng::from_rng(thread_rng()).unwrap();

        let uniforms = Uniform::new(0, self.k);
        for doc in self.docs.iter_mut() {
            for token in doc {
                let t = uniforms.sample(&mut rng);
                token.t = t;
                self.cdk[token.d][t] += 1;
                self.cd[token.d] += 1;
                self.ckv[token.w][t] += 1;
                self.ck[t] += 1;
            }
        }

        sample_lda(
            &mut self.docs,
            &mut self.cdk,
            &mut self.ckv,
            &mut self.cd,
            &mut self.ck,
            self.k,
            self.v,
            self.alpha,
            self.beta,
            2000,
            true,
            true,
            &mut rng,
        );

        // for t in  0..self.k {
        //     let mut vals = self.ckv.iter().map(|c| c[t]).zip(0..self.v).collect::<Vec<_>>();
        //     vals.sort_by_key(|(t, _)| *t);
        //     let vals = vals.iter().rev().take(10).map(|(val, index)| (self.vocab[*index].clone(), *val)).collect::<Vec<_>>();
        //     println!("topic {}: {:?}", t, vals);
        // }
    }

    fn get_theta(&self) -> Vec<Vec<f64>> {
        let mut theta = Vec::from_iter((0..self.docs.len()).map(|_| vec![0f64; self.k]));
        for d in 0..self.docs.len() {
            for t in 0..self.k {
                theta[d][t] = (self.cdk[d][t] as f64 + self.alpha)
                    / (self.cd[d] as f64 + self.k as f64 * self.alpha);
            }
        }
        theta
    }

    fn infer_theta(&self, doc: Document, rng: &mut XorShiftRng) -> Vec<f64> {
        let mut cdk = vec![vec![0usize; self.k]];
        let mut cd = vec![0usize; 1];
        let mut ckv = self.ckv.clone();
        let mut ck = self.ck.clone();
        let uniforms = Uniform::new(0, self.k);
        let mut docs = vec![Vec::from_iter(doc.token.iter().filter_map(|token| {
            if self.vocab_map.contains_key(token) {
                let t = uniforms.sample(rng);
                cdk[0][t] += 1;
                cd[0] += 1;
                let w = self.vocab_map[token];
                ckv[w][t] += 1;
                ck[t] += 1;

                Some(LDADoc {
                    d: 0,
                    w,
                    t
                })
            } else { None }
        }))];

        sample_lda(
            &mut docs,
            &mut cdk,
            &mut ckv,
            &mut cd,
            &mut ck,
            self.k,
            self.v,
            self.alpha,
            self.beta,
            500,
            false,
            true,
            rng,
        );

        let mut theta = vec![0f64; self.k];
        for t in 0..self.k {
            theta[t] =
                (cdk[0][t] as f64 + self.alpha) / (self.cd[0] as f64 + self.k as f64 * self.alpha);
        }
        theta
    }
}

#[derive(Clone, Debug)]
struct LDADoc {
    d: usize,
    w: usize,
    t: usize,
}

fn sample_lda(
    docs: &mut Vec<Vec<LDADoc>>,
    cdk: &mut Vec<Vec<usize>>,
    ckv: &mut Vec<Vec<usize>>,
    cd: &mut Vec<usize>,
    ck: &mut Vec<usize>,
    k: usize,
    v: usize,
    alpha: f64,
    beta: f64,
    samples: usize,
    debug: bool,
    shuffle: bool,
    rng: &mut XorShiftRng,
) {
    let mut last_changes = usize::MAX;
    let mut changes = 0;
    let mut debug_changes = 0;
    let mut it_since_reset = 0;
    for it in 0..samples {
        if shuffle {
            docs.shuffle(rng);
        }
        for doc in docs.iter_mut() {
            if shuffle {
                doc.shuffle(rng);
            }
            for token in doc.iter_mut() {
                cdk[token.d][token.t] -= 1;
                ckv[token.w][token.t] -= 1;
                cd[token.d] -= 1;
                ck[token.t] -= 1;
                let p = get_cond_prob(token.d, token.w, cdk, cd, ckv, ck, k, v, alpha, beta);
                let t = WeightedIndex::new(p).unwrap().sample(rng);
                if t != token.t {
                    debug_changes += 1;
                    changes += 1
                }
                token.t = t;
                cdk[token.d][token.t] += 1;
                ckv[token.w][token.t] += 1;
                cd[token.d] += 1;
                ck[token.t] += 1;
            }
        }
        if it % 100 == 0 {
            let rel_change = changes as f64 / (last_changes as f64);
            if rel_change > 0.99 && rel_change < 1.05 {
                if debug {
                    println!("early break due to rel change {rel_change} at iteration {it}");
                }
                break;
            }
        }
        it_since_reset += 1;
        if debug && it % 20 == 0 {
            println!(" iteration {it} of {samples}, average changes {}", debug_changes as f64 / it_since_reset as f64);
            it_since_reset = 0;
            debug_changes = 0;
        }
    }
}

fn get_cond_prob(
    d: usize,
    w: usize,
    cdk: &Vec<Vec<usize>>,
    cd: &Vec<usize>,
    ckv: &Vec<Vec<usize>>,
    ck: &Vec<usize>,
    k: usize,
    v: usize,
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut p = vec![0f64; k];
    let d1 = k as f64 * alpha + cd[d] as f64;
    let d2 = v as f64 * beta;
    for t in 0..k {
        let l = (cdk[d][t] as f64 + alpha) / d1;
        let r = (ckv[w][t] as f64 + beta) / (d2 + ck[t] as f64);
        p[t] = l * r;
    }
    p
}

#[derive(Deserialize, Clone)]
struct Document {
    id: String,
    token: Vec<String>,
    author: String,
}

impl Document {
    fn merge_docs(docs: Vec<Document>) -> Vec<Document> {
        let mut doc_map = HashMap::new();
        for doc in docs {
            doc_map
                .entry(doc.author.clone())
                .or_insert(Document {
                    id: doc.id,
                    token: vec![],
                    author: doc.author.clone(),
                })
                .token
                .extend(doc.token.into_iter());
        }
        doc_map.into_values().collect()
    }
}

fn hellinger(theta1: &Vec<f64>, theta2: &Vec<f64>) -> f64 {
    let k = theta1.len();
    let f = 1.0 / (2.0f64.sqrt());
    let mut sum = 0f64;
    for t in 0..k {
        sum += (theta1[t].sqrt() - theta2[t].sqrt()).powi(2);
    }
    sum = sum.sqrt();
    f * sum
}
