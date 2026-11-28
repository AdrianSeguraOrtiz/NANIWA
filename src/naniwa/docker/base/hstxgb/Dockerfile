FROM --platform=linux/amd64 python@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

LABEL org.opencontainers.image.source="https://github.com/MingqingHuang-SHU/HSTXGB" \
      org.opencontainers.image.revision="689705a91a7f40ea39f15e6f8bdcd706c5c32fa0" \
      org.opencontainers.image.description="ANDREA HSTXGB expression-only integration; pinned native hyperparameter search and split-count scoring"

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1=14.2.0-19 \
    && rm -rf /var/lib/apt/lists/*
# Upstream specifies no package versions. This tested CPU-only dependency set
# fixes the integration runtime; it is not an asserted author environment.
RUN printf '%s\n' \
      'numpy==1.26.4 --hash=sha256:666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5' \
      'scipy==1.12.0 --hash=sha256:8b8066bce124ee5531d12a74b617d9ac0ea59245246410e19bca549656d9a40a' \
      'scikit-learn==1.4.2 --hash=sha256:44c62f2b124848a28fd695db5bc4da019287abf390bfce602ddc8aa1ec186aae' \
      'joblib==1.4.2 --hash=sha256:06d478d5674cbc267e7496a410ee875abd68e4340feff4490bcb7afb88060ae6' \
      'threadpoolctl==3.5.0 --hash=sha256:56c1e26c150397e58c4926da8eeee87533b1e32bef131bd4bf6a2f45f3185467' \
      'xgboost-cpu==2.1.4 --hash=sha256:ac3281127b2f77f6c62e8699216b0974a1c4a442ee6162eb94256c01df7cd1f5' \
    | python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /dev/stdin
COPY run_tool.py /app/run_tool.py
RUN python /app/run_tool.py --prepare-upstream --output-dir /app
COPY run_tool.sh /app/run_tool.sh
COPY _run_tool_common.py /app/_run_tool_common.py
COPY _column_metadata.py /app/_column_metadata.py
RUN chmod +x /app/run_tool.sh
ENTRYPOINT ["/app/run_tool.sh"]
