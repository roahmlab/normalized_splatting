# Submodule patches

`simple-knn` is an upstream Inria submodule we cannot push to, but it does not
build against newer CUDA/GCC toolchains without one extra include: `FLT_MAX` is
used in `simple_knn.cu` while `<cfloat>` is never included. Older toolchains
pulled it in transitively; current ones do not.

Apply it after checking out submodules:

```bash
git submodule update --init --recursive
git -C submodules/simple-knn apply ../../patches/simple-knn-cfloat.patch
```

If the build already succeeds without it, skip it.
