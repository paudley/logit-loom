# SPDX-License-Identifier: MIT OR Apache-2.0

.PHONY: check check-core doc models-check package package-list release-check clean

check: models-check
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --locked -- -D warnings
	cargo test --workspace --all-targets --locked
	cargo test --workspace --doc --locked

check-core: models-check
	cargo fmt --all --check
	cargo clippy -p logit-loom-models -p logit-loom-core -p logit-loom-executor -p logit-loom -p logit-loom-diffusion -p logit-loom-tokenizer --all-targets --locked -- -D warnings
	cargo test -p logit-loom-models -p logit-loom-core -p logit-loom-executor -p logit-loom -p logit-loom-diffusion -p logit-loom-tokenizer --all-targets --locked
	cargo test -p logit-loom-models -p logit-loom-core -p logit-loom-executor -p logit-loom -p logit-loom-diffusion -p logit-loom-tokenizer --doc --locked

models-check:
	cargo run --quiet --locked -p logit-loom-xtask -- models check

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked

package:
	cargo package -p logit-loom-core --locked
	cargo package -p logit-loom-executor --locked
	cargo package -p logit-loom-models --locked
	cargo package -p logit-loom --locked
	cargo package -p logit-loom-diffusion --locked
	cargo package -p logit-loom-llamacpp --locked
	cargo package -p logit-loom-runtime --locked
	cargo package -p logit-loom-diffusion-sdcpp --locked

package-list:
	cargo package -p logit-loom-core --allow-dirty --list
	cargo package -p logit-loom-executor --allow-dirty --list
	cargo package -p logit-loom-models --allow-dirty --list
	cargo package -p logit-loom --allow-dirty --list
	cargo package -p logit-loom-diffusion --allow-dirty --list
	cargo package -p logit-loom-llamacpp --allow-dirty --list
	cargo package -p logit-loom-runtime --allow-dirty --list
	cargo package -p logit-loom-diffusion-sdcpp --allow-dirty --list

release-check:
	scripts/release-check.sh

clean:
	cargo clean
