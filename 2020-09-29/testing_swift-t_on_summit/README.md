# Problem

Swift/T does not seem to work on Summit.

Can show this by running `bsub test.sh`.

Sample output is at Not_Specified.375996.

## Scheduled

From login node:
```
$ ./test-2.sh
```

## Interactive

From login node:
```
$ ./go-interactive.sh
```

Once compute node session starts:
```
$ ./test-interactive-2.sh
```

## Interactive node call

```bash
bsub -W 01:00 -nnodes 2 -P med106 -q debug -Is /bin/bash
```
