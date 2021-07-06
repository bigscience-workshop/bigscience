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

# Network Topology

V100 32GB GPU are `r6i[4-7]n[0-8],r[7-9]i[0-7]n[0-8],r14i7n[0-8]`

They are mostly grouped together but that doesn't really mean that the switches are completely independent from the rest of the network.

Due to the hypercube topology used on JZ reaching two nodes on different racks might use intermediate hops on other racks. e.g. communications between nodes on r6 and r7 might go through switches on r3 or r8 depending of the targeted nodes.
