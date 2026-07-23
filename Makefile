# SPDX-License-Identifier: MIT OR Apache-2.0

.PHONY: check check-core doc package package-list release-check clean

check:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --locked -- -D warnings
	cargo test --workspace --all-targets --locked
	cargo test --workspace --doc --locked

check-core:
	cargo fmt --all --check
	cargo clippy -p logit-loom-core -p logit-loom --all-targets --locked -- -D warnings
	cargo test -p logit-loom-core -p logit-loom --all-targets --locked
	cargo test -p logit-loom-core -p logit-loom --doc --locked

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked

package:
	cargo package -p logit-loom-core --locked
	cargo package -p logit-loom --locked
	cargo package -p logit-loom-llamacpp --locked

package-list:
	cargo package -p logit-loom-core --allow-dirty --list
	cargo package -p logit-loom --allow-dirty --list
	cargo package -p logit-loom-llamacpp --allow-dirty --list

release-check:
	scripts/release-check.sh

clean:
	cargo clean
