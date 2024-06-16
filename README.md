# autodidax-ocaml

An WIP attempt at porting [Autodidax: JAX core from scratch](https://jax.readthedocs.io/en/latest/autodidax.html#part-2-jaxprs) to OCaml.

## notes

Depends on [LaurentMazare/ocaml-xla](https://github.com/LaurentMazare/ocaml-xla),
which isn't distributed via opam so you'll need to build + install locally.
It also needs some [minor fixes](https://github.com/mt-caret/ocaml-xla).
When building/depending on ocaml-xla on macos, we get an error like

```
dyld[66252]: Library not loaded: bazel-out/darwin_arm64-opt/bin/tensorflow/compiler/xla/extension/libxla_extension.so
...
```

Solution here is to:

```
DYLD_LIBRARY_PATH=/absolute/path/to/directory-with-libxla-extension-so/ dune runtest/build/etc.
```