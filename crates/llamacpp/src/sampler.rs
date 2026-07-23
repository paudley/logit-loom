// SPDX-License-Identifier: MIT OR Apache-2.0

//! Translation from backend-neutral plans to native llama.cpp samplers.

use std::panic::{AssertUnwindSafe, catch_unwind};

use llama_cpp_4::model::LlamaModel;
use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_4::token::LlamaToken;
use logit_loom::{GenerationPlan, LogitBias, MirostatVersion, TokenId};

use crate::Error;

pub(crate) fn build_sampler(
    model: &LlamaModel,
    plan: &GenerationPlan,
    history: &[TokenId],
) -> Result<LlamaSampler, Error> {
    plan.validate()?;
    validate_bias_tokens(model.n_vocab(), &plan.biases)?;
    catch_unwind(AssertUnwindSafe(|| {
        build_sampler_inner(model, plan, history)
    }))
    .map_err(|payload| {
        let message = payload.downcast_ref::<&str>().map_or_else(
            || {
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_else(|| "non-string panic payload".to_owned())
            },
            |value| (*value).to_owned(),
        );
        Error::Native(format!("native sampler construction panicked: {message}"))
    })?
}

fn build_sampler_inner(
    model: &LlamaModel,
    plan: &GenerationPlan,
    history: &[TokenId],
) -> Result<LlamaSampler, Error> {
    let mut stages = Vec::new();
    if let Some(grammar) = &plan.grammar {
        stages.push(LlamaSampler::grammar(model, &grammar.source, &grammar.root));
    }
    if !plan.biases.is_empty() {
        let biases = plan
            .biases
            .iter()
            .map(|bias| (LlamaToken::new(bias.token.get()), bias.bias))
            .collect::<Vec<_>>();
        stages.push(LlamaSampler::logit_bias(model.n_vocab(), &biases));
    }
    if let Some(repetition) = &plan.sampling.repetition {
        let mut sampler = LlamaSampler::penalties(
            repetition.last_n,
            repetition.repeat_penalty,
            repetition.frequency_penalty,
            repetition.presence_penalty,
        );
        accept_history(&mut sampler, history);
        stages.push(sampler);
    }
    if let Some(dry) = &plan.sampling.dry {
        let training_context = i32::try_from(model.n_ctx_train())
            .map_err(|_| Error::Incompatible("model training context exceeds i32".to_owned()))?;
        let mut sampler = LlamaSampler::new().dry(
            model,
            training_context,
            dry.multiplier,
            dry.base,
            dry.allowed_length,
            dry.penalty_last_n,
            &dry.sequence_breakers,
        );
        accept_history(&mut sampler, history);
        stages.push(sampler);
    }
    if plan.sampling.top_k > 0 {
        stages.push(LlamaSampler::top_k(plan.sampling.top_k));
    }
    if plan.sampling.typical_p < 1.0 {
        stages.push(LlamaSampler::typical(plan.sampling.typical_p, 1));
    }
    if plan.sampling.top_p < 1.0 {
        stages.push(LlamaSampler::top_p(plan.sampling.top_p, 1));
    }
    if plan.sampling.min_p > 0.0 {
        stages.push(LlamaSampler::min_p(plan.sampling.min_p, 1));
    }

    if let Some(mirostat) = &plan.sampling.mirostat {
        if plan.sampling.temperature > 0.0 {
            stages.push(LlamaSampler::temp(plan.sampling.temperature));
        }
        stages.push(match mirostat.version {
            MirostatVersion::V1 => LlamaSampler::mirostat(
                model.n_vocab(),
                plan.sampling.seed,
                mirostat.tau,
                mirostat.eta,
                mirostat.m,
            ),
            MirostatVersion::V2 => {
                LlamaSampler::mirostat_v2(plan.sampling.seed, mirostat.tau, mirostat.eta)
            }
        });
    } else if plan.sampling.temperature == 0.0 {
        stages.push(LlamaSampler::greedy());
    } else {
        stages.push(LlamaSampler::temp(plan.sampling.temperature));
        stages.push(LlamaSampler::dist(plan.sampling.seed));
    }

    Ok(LlamaSampler::chain_simple(stages))
}

fn accept_history(sampler: &mut LlamaSampler, history: &[TokenId]) {
    sampler.accept_many(history.iter().map(|token| LlamaToken::new(token.get())));
}

fn validate_bias_tokens(n_vocab: i32, biases: &[LogitBias]) -> Result<(), Error> {
    if n_vocab <= 0 {
        return Err(Error::Incompatible(
            "model vocabulary must be positive".to_owned(),
        ));
    }
    if let Some(bias) = biases.iter().find(|bias| bias.token.get() >= n_vocab) {
        return Err(Error::Invalid(format!(
            "logit bias token {} is outside vocabulary size {n_vocab}",
            bias.token.get()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_bias_tokens_must_fit_the_loaded_vocabulary() {
        let inside = LogitBias {
            token: TokenId::new(9).unwrap(),
            bias: 1.0,
        };
        let outside = LogitBias {
            token: TokenId::new(10).unwrap(),
            bias: 1.0,
        };
        assert!(validate_bias_tokens(10, &[inside]).is_ok());
        assert!(validate_bias_tokens(10, &[outside]).is_err());
        assert!(validate_bias_tokens(0, &[]).is_err());
    }
}
