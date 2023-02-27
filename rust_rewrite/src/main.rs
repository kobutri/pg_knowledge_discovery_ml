use anyhow::*;
use rand::{
    distributions::{Uniform, WeightedIndex},
    prelude::*,
};
use std::{collections::HashMap, hash::Hash, ops::Index, slice::SliceIndex};

#[derive(Debug, serde::Deserialize)]
struct InputRecord {
    id: String,
    text: Vec<String>,
    author: String,
}

fn read(path: &str) -> Vec<InputRecord> {
    let buffer = std::fs::read_to_string(path).unwrap();
    return serde_json::from_str(&buffer).unwrap();
}

#[derive(Default, Clone)]
struct OrderedSet<T: Clone + Eq + Hash> {
    items: Vec<T>,
    items_map: HashMap<T, usize>,
}

impl<T: Clone + Eq + Hash> OrderedSet<T> {
    fn add(&mut self, id: &T) -> usize {
        if let Some(index) = self.items_map.get(id) {
            return *index;
        } else {
            let index = self.items.len();
            self.items.push(id.to_owned());
            self.items_map
                .insert(self.items.last().unwrap().to_owned(), index);
            return index;
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl<T: Clone + Eq + Hash, Idx> Index<Idx> for OrderedSet<T>
where
    Idx: SliceIndex<[T], Output = T>,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: Idx) -> &Self::Output {
        self.items.index(index)
    }
}

#[derive(Default, Clone, Debug)]
struct Token {
    token: usize,
    author: usize,
    doc_id: usize,
}

#[derive(Clone, Copy)]
struct Params {
    k: usize,
    alpha: f32,
    beta: f32,
    iteration_count: usize,
    cut_off: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self::new(10, 10, 5e-4)
    }
}

impl Params {
    fn new(k: usize, iteration_count: usize, cut_off: f32) -> Self {
        let alpha: f32 = 0.1f32.min(50.0 / k as f32);
        let beta: f32 = 0.01;

        Self {
            k,
            alpha,
            beta,
            iteration_count,
            cut_off,
        }
    }
}

#[derive(Default, Clone)]
struct Model {
    vocab: OrderedSet<String>,
    authors: OrderedSet<String>,
    ids: OrderedSet<String>,
    tokens: Vec<Token>,
    z: Vec<usize>,
    cdk: Vec<Vec<usize>>,
    cd: Vec<usize>,
    ckv: Vec<Vec<usize>>,
    ck: Vec<usize>,
    params: Params,
}

impl Model {
    fn new(input: &Vec<InputRecord>, params: Params) -> Self {
        // let binding = input.iter().fold(HashMap::new(), |mut acc, value| {
        //     let text= &mut acc.entry(&value.author).or_insert(InputRecord {
        //         text: vec![],
        //         id: value.id.clone(),
        //         author: value.author.clone()
        //     }).text;
        //     for token in &value.text {
        //         text.push(token.clone());
        //     }

        //     acc
        // });
        // let input = binding.into_values().collect::<Vec<_>>();
        // let input = &input;

        let mut high_pass = HashMap::new();
        for doc in input {
            for token in &doc.text {
                *high_pass.entry(token).or_insert(0) += 1;
            }
        }

        let mut vocab = OrderedSet::default();
        let mut ids = OrderedSet::default();
        let mut authors = OrderedSet::default();
        let mut tokens = vec![];

        for doc in input {
            let doc_id = ids.add(&doc.id);
            let author = authors.add(&doc.author);
            for token in &doc.text {
                if *high_pass.get(token).unwrap() as f32 > params.cut_off * input.len() as f32 {
                    let token = vocab.add(token);
                    tokens.push(Token {
                        author,
                        token,
                        doc_id,
                    });
                }
            }
        }

        let mut rng = rand_xorshift::XorShiftRng::from_rng(thread_rng()).unwrap();
        let z = Uniform::from(0..params.k)
            .sample_iter(&mut rng)
            .take(tokens.len())
            .collect::<Vec<_>>();

        let mut cdk = (0..tokens.len())
            .map(|_| vec![0usize; params.k])
            .collect::<Vec<_>>();
        let mut ckv = (0..vocab.len())
            .map(|_| vec![0usize; params.k])
            .collect::<Vec<_>>();
        let mut cd = vec![0; tokens.len()];
        let mut ck = vec![0; params.k];

        for i in 0..z.len() {
            cdk[tokens[i].doc_id][z[i]] += 1;
            ckv[tokens[i].token][z[i]] += 1;
            cd[tokens[i].doc_id] += 1;
            ck[z[i]] += 1;
        }

        let mut this = Self {
            vocab,
            authors,
            ids,
            tokens,
            z,
            cdk,
            cd,
            ckv,
            ck,
            params,
        };

        this.train(true);

        return this;
    }

    fn train(&mut self, log: bool) {
        let mut rng = rand_xorshift::XorShiftRng::from_rng(thread_rng()).unwrap();
        let topic_options = (0..self.params.k).collect::<Vec<_>>();
        for it in 0..self.params.iteration_count {
            let mut changed = 0;
            for i in 0..self.tokens.len() {
                self.cdk[self.tokens[i].doc_id][self.z[i]] -= 1;
                self.ckv[self.tokens[i].token][self.z[i]] -= 1;
                self.cd[self.tokens[i].doc_id] -= 1;
                self.ck[self.z[i]] -= 1;
                let p = {
                    let mut ps = vec![];
                    for t in 0..self.params.k {
                        let num1 = self.params.alpha + self.cdk[self.tokens[i].doc_id][t] as f32;
                        let denom1 = self.params.k as f32 * self.params.alpha
                            + self.cd[self.tokens[i].doc_id] as f32;
                        let num2 = self.params.beta + self.ckv[self.tokens[i].token][t] as f32;
                        let denom2 = self.vocab.len() as f32 * self.params.beta + self.ck[t] as f32;
                        let p = (num1 / denom1) * (num2 / denom2);
                        ps.push(p);
                    }
                    let sum: f32 = ps.iter().sum();
                    let ps = ps.iter().map(|p| p / sum).collect::<Vec<_>>();
                    ps
                };
                let dist = WeightedIndex::new(&p).unwrap();
                let new_topic = dist.sample(&mut rng);
                if new_topic != self.z[i] {
                    changed += 1;
                }
                self.z[i] = new_topic;
                self.cdk[self.tokens[i].doc_id][self.z[i]] += 1;
                self.ckv[self.tokens[i].token][self.z[i]] += 1;
                self.cd[self.tokens[i].doc_id] += 1;
                self.ck[self.z[i]] += 1;
            }
            if log {
                println!(
                    "iteration {} finished, changed {} assignments",
                    it + 1,
                    changed
                );
            }
        }

        if log {
            for i in 0..self.params.k {
                let mut topic = self
                    .ckv
                    .iter()
                    .enumerate()
                    .map(|(index, v)| (&self.vocab.items[index], v[i] as f32 / (self.ck[i] as f32)))
                    .collect::<Vec<_>>();
                topic.sort_by(|(_, v1), (_, v2)| v1.total_cmp(v2));
                topic.reverse();
                println!("{:?}", &topic[0..10])
            }
        }
    }

    fn predict(&self, input: &InputRecord) -> String {
        let mut this = self.clone();
        this.authors = OrderedSet::default();
        this.ids = OrderedSet::default();
        this.tokens = vec![];

        //let author_id = this.authors.add(&input.author);
        let doc_id = this.ids.add(&"unknown".to_string());

        for token in &input.text {
            if this.vocab.items_map.contains_key(token) {
                let token_id = this.vocab.add(&token);
                this.tokens.push(Token {
                    token: token_id,
                    author: 0,
                    doc_id,
                });
            }
        }

        let mut rng = rand_xorshift::XorShiftRng::from_rng(thread_rng()).unwrap();
        this.z = Uniform::from(0..this.params.k)
            .sample_iter(&mut rng)
            .take(this.tokens.len())
            .collect::<Vec<_>>();

        this.cdk = (0..this.tokens.len())
            .map(|_| vec![0usize; this.params.k])
            .collect::<Vec<_>>();

        this.cd = vec![0; this.tokens.len()];

        if self.vocab.len() != this.vocab.len() {
            let diff = this.vocab.len() - self.vocab.len();
            for _ in 0..diff {
                this.ckv.push(vec![0; this.params.k]);
            }
            for _ in 0..diff {
                this.ck.push(0);
            }
        }

        for i in 0..this.z.len() {
            this.cdk[this.tokens[i].doc_id][this.z[i]] += 1;
            this.ckv[this.tokens[i].token][this.z[i]] += 1;
            this.cd[this.tokens[i].doc_id] += 1;
            this.ck[this.z[i]] += 1;
        }

        this.train(false);

        let theta1 = this.theta();
        let theta2 = self.theta();

        theta1
            .iter()
            .zip(theta2)
            .map(|(theta1, theta2)| {
                (0.5f32
                    * theta1
                        .iter()
                        .zip(theta2)
                        .map(|(theta1, theta2)| (theta1.sqrt() - theta2.sqrt()).powi(2))
                        .sum::<f32>())
                .sqrt()
            })
            .enumerate()
            .fold(HashMap::new(), |mut authors, (index, diff)| {
                let entry = &mut *authors
                    .entry(&self.authors[self.tokens[index].author])
                    .or_insert((0f32, 0usize));
                entry.0 += diff;
                entry.1 += 1;
                authors
            })
            .iter()
            .map(|(key, (val, sum))| (val / (*sum as f32), *key))
            .max_by(|(val, _), (val2, _)| val.total_cmp(val2))
            .unwrap()
            .1
            .to_owned()
    }

    fn theta(&self) -> Vec<Vec<f32>> {
        self.cdk
            .iter()
            .zip(&self.cd)
            .map(|(cdk, sum)| {
                cdk.iter()
                    .map(|value| {
                        (*value as f32 + self.params.alpha)
                            / (*sum as f32 + self.params.k as f32 * self.params.alpha)
                    })
                    .collect()
            })
            .collect()
    }
}

#[derive(Default, Debug)]
struct PredictionResult {
    t_p: usize,
    f_n: usize,
    f_p: usize,
    expected: usize,
}

fn accuracy(result: &PredictionResult, total: usize) -> f32 {
    let tn = total - result.t_p + result.f_p;
    return (tn + result.t_p) as f32 / (total as f32);
}

fn main() -> Result<()> {
    let train_input = read("tokenized_test.json");
    let model = Model::new(&train_input, Params::new(50, 200, 5e-4));

    let test_input = read("tokenized_test.json");

    let predictions = &test_input[0..300]
        .into_iter()
        .enumerate()
        .map(|(index, input)| {
            if index % 101 == 1 {
                println!("test doc {} complete", index - 1);
            }
            (model.predict(&input), input)
        })
        .collect::<Vec<_>>();
    let results = predictions
        .iter()
        .fold(HashMap::new(), |mut map, (prediction, input)| {
            map.entry(&input.author)
                .or_insert(PredictionResult::default())
                .expected += 1;
            if *prediction == *input.author {
                map.get_mut(&input.author).unwrap().t_p += 1;
            } else {
                map.get_mut(&input.author).unwrap().f_n += 1;
                map.entry(&prediction)
                    .or_insert(PredictionResult::default())
                    .f_p += 1;
            }
            map
        });
    println!("{:#?}", results);

    Ok(())
}
