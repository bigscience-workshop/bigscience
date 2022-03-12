# Specs of Jean Zay

- 261 nodes, with V100 32 GB GPUs: total 1044 GPUs
- 351 nodes, with V100 16 GB GPUs: total 1404 GPUs

## Disc Partitions

- `$HOME` - 3GB for small files
- `$WORK` - 5TB / 500k inodes → sources, input/output files
- `$SCRATCH` - fastest (full SSD), 400TB our quota (total 2PB), files auto-removed after 30 days without access
- `$STORE` - for long term storage in tar files (very few inodes!)

## Shared Filesystem

- GPFS filesystem (Spectrum Scale)

- `$SCRATCH` - is SSD with theoretical bandwidth of at least 300 GB/s, probably more with the 2PB extension
- other partitions are slower discs

## Network Topology

V100 32GB GPU are `r6i[4-7]n[0-8],r[7-9]i[0-7]n[0-8],r14i7n[0-8]`

They are mostly grouped together but that doesn't really mean that the switches are completely independent from the rest of the network.

Due to the hypercube topology used on JZ reaching two nodes on different racks might use intermediate hops on other racks. e.g. communications between nodes on r6 and r7 might go through switches on r3 or r8 depending of the targeted nodes.

## JZ3

coming in Jan 2022:

- GPUs: 416 A100 80GB GPUs (52 nodes of 8 gpus each)
- 8 GPUs per node Using NVLink 4 inter-gpu connects, 4 OmniPath links
- CPU: AMD
- CPU memory: 512GB per node
- Inter-node connect: Omni-Path Architecture (OPA)
- NCCL-communications network: a fully dedicated subnet
- Disc IO network: shared network with other types of nodes
