use std::collections::HashMap;

use rand::{distributions::{Uniform, WeightedIndex}, prelude::Distribution, thread_rng, SeedableRng};

fn main() {
    let train_file = std::fs::read_to_string("./tokenized_train.json").unwrap();
    let train_data: serde_json::Value = serde_json::from_str(&train_file).unwrap();

    let mut vocab_freq_map = HashMap::new();
    let mut token_count = 0;
    for doc in train_data.as_array().unwrap() {
        for token in doc["text"].as_array().unwrap() {
            token_count += 1;
            *vocab_freq_map.entry(token.as_str().unwrap()).or_insert(0) += 1;
        }
    }

    let mut vocab_map = HashMap::new();
    let mut vocab = vec![];

    for (key, value) in vocab_freq_map {
        if (value as f64 / token_count as f64) > 1e-5 {
            if !vocab_map.contains_key(key) {
                vocab_map.insert(key, vocab.len());
                vocab.push(key);
            }
        }
    }

    println!("vocab size: {}", vocab.len());

    let mut docs = vec![];
    let mut authors = vec![];
    let mut authors_map = HashMap::new();

    let mut d_id = vec![];
    let mut a_id = vec![];
    let mut t_id = vec![];

    for (i, doc) in train_data.as_array().unwrap().iter().enumerate() {
        docs.push(doc["id"].as_str().unwrap());

        let mut author_id = 0;
        if authors_map.contains_key(doc["author"].as_str().unwrap()) {
            author_id = *authors_map.get(doc["author"].as_str().unwrap()).unwrap();
        } else {
            authors_map.insert(doc["author"].as_str().unwrap(), authors.len());
            author_id = authors.len();
            authors.push(doc["author"].as_str().unwrap());
        }
        for token in doc["text"].as_array().unwrap() {
            if vocab_map.contains_key(token.as_str().unwrap()) {
                d_id.push(i);
                a_id.push(author_id);
                t_id.push(*vocab_map.get(token.as_str().unwrap()).unwrap());
            }
        }
    }

    let D = docs.len();
    let N = t_id.len();
    let K = 20usize;
    let V = vocab.len();
    let alpha = 0.1;
    let beta = 0.001;
    dbg!(N, K, V, alpha, beta);

    let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(0);

    let mut Z = Uniform::new(0, K)
        .sample_iter(&mut rng)
        .take(N)
        .collect::<Vec<_>>();

    let mut cdk = vec![];
    let mut ckv = vec![];
    let mut cd = vec![0usize; D];
    let mut ck = vec![0usize; K];

    for _ in 0..K {
        let v1 = vec![0usize; N];
        cdk.push(v1);
        let v2 = vec![0usize; V];
        ckv.push(v2);
    }

    for i in 0..N {
        cd[d_id[i]] += 1;
        cdk[Z[i]][d_id[i]] += 1;
        ckv[Z[i]][t_id[i]] += 1;
        ck[Z[i]] += 1;
    }

    for it in 0..1000 {
        let mut changes = 0;
        for i in 0..N {
            cdk[Z[i]][d_id[i]] -= 1;
            ckv[Z[i]][t_id[i]] -= 1;
            cd[d_id[i]] -= 1;
            ck[Z[i]] -= 1;

            let mut ps = vec![];
            let mut p_sum = 0f64;
            for j in 0..K {
                let num1 = alpha + cdk[j][d_id[i]] as f64;
                let denom1 = K as f64 * alpha + cd[d_id[i]] as f64;
                let num2 = beta + ckv[j][t_id[i]] as f64;
                let denom2 = V as f64 * beta + ck[j] as f64;
                let p_temp = (num1 * num2) / (denom1 * denom2);
                p_sum += p_temp;
                ps.push(p_temp);
            }

            let dist = WeightedIndex::new(ps).unwrap();
            let new_topic = dist.sample(&mut rng);
            if new_topic != Z[i] {
                changes += 1;
            }

            Z[i] = new_topic;
            cdk[Z[i]][d_id[i]] += 1;
            ckv[Z[i]][t_id[i]] += 1;
            cd[d_id[i]] += 1;
            ck[Z[i]] += 1;
        }
        println!("iteration {} complete, changed {} assignments", it, changes);
    }

    let result = serde_json::json!({
        "docs": docs, 
        "vocab": vocab, 
        "authors": authors,
        "Z": Z,
        "d_id": d_id,
        "a_id": a_id,
        "t_id" : t_id,
    });
    std::fs::write("./result.json", serde_json::to_string_pretty(&result).unwrap()).unwrap();
}
