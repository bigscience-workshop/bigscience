# Specs of Jean Zay


# Disc Partitions

- `$HOME` - 3GB for small files
- `$WORK` - 5TB / 500k inodes → sources, input/output files
- `$SCRATCH` - fastest (full SSD), no quota (2PB), files auto-removed after 30 days without access
- `$STORE` - for long term storage in tar files (very few inodes!)

# Shared Filesystem

- GPFS filesystem (Spectrum Scale)

- `$SCRATCH` - is SSD with theoretical bandwidth of at least 300 GB/s, probably more with the 2PB extension
- other partitions are slower discs
