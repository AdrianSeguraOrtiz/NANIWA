FROM --platform=linux/amd64 python@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

LABEL org.opencontainers.image.source="https://github.com/CSUBioGroup/NIMCE" \
      org.opencontainers.image.revision="095fdb20aaa4748f31c2d728bec5920f395156be" \
      org.opencontainers.image.description="ANDREA NIMCE integration using the pinned original MATLAB implementation under GNU Octave"

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       octave=9.4.0-1 octave-statistics=1.7.3-2 libopenblas0-pthread=0.3.29+ds-3 \
    && dpkg-query -W > /app/system-packages.tsv \
    && rm -rf /var/lib/apt/lists/*
RUN printf '%s\n' \
      'numpy==1.26.4 --hash=sha256:666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5' \
    | python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /dev/stdin
COPY run_tool.py /app/run_tool.py
# Source preparation downloads and verifies the public commit; local evidence
# directories never enter the ordinary build context.
RUN python /app/run_tool.py --prepare-upstream --output-dir /app/nimce
COPY run_tool.sh /app/run_tool.sh
COPY _run_tool_common.py /app/_run_tool_common.py
COPY _column_metadata.py /app/_column_metadata.py
RUN chmod +x /app/run_tool.sh
ENTRYPOINT ["/app/run_tool.sh"]
