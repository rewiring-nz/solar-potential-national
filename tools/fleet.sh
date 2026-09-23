#!/bin/bash
# The fleet: N spot VMs that each run src/worker.sh until the queue is empty
# and then delete themselves.
#
#   tools/fleet.sh image          # snapshot the build VM's disk as the worker image
#   tools/fleet.sh up 6           # start 6 workers
#   tools/fleet.sh status         # queue counts + who is building what
#   tools/fleet.sh list           # the worker VMs
#   tools/fleet.sh down           # delete every worker VM
#
# WHY C2D IN ZONE -a. T2D quota in Sydney is 24 cores and the build VM holds
# 16 of them; C2D has 100 free and c2d-standard-16 exists only in -a
# (measured 22 Sep). The regional CPU ceiling is 100, so six 16-core workers
# is the most the project can run today without a quota request; past that,
# request "C2D CPUs" for australia-southeast1 in the console.
#
# SCOPES. The build VM's service account carries storage READ-ONLY scope,
# fixed at creation, so it cannot publish. Workers are created with
# storage-rw (to publish) and compute-rw (to delete themselves).
set -u
ZONE=${FLEET_ZONE:-australia-southeast1-a}
SRC_VM=${FLEET_SOURCE_VM:-claude-doing-things}
SRC_ZONE=${FLEET_SOURCE_ZONE:-australia-southeast1-b}
IMAGE=${FLEET_IMAGE:-solar-worker}
MACHINE=${FLEET_MACHINE:-c2d-standard-16}
DISK_GB=${FLEET_DISK_GB:-400}
PREFIX=solar-worker
STARTUP=$(cat <<'EOS'
#!/bin/bash
# runs as root at boot; the checkout and venv live in j's home on the image
su - j -c 'cd ~/solar-map && git pull -q 2>/dev/null; SOLAR_SELF_DELETE=1 ./src/worker.sh' \
  >> /var/log/solar-worker.log 2>&1
EOS
)

case "${1:-}" in
  image)
    echo "imaging $SRC_VM's disk as $IMAGE (the VM keeps running; --force allows that)"
    gcloud compute images delete "$IMAGE" --quiet 2>/dev/null || true
    gcloud compute images create "$IMAGE" --source-disk "$SRC_VM" \
      --source-disk-zone "$SRC_ZONE" --force --quiet
    ;;
  up)
    N=${2:?how many workers}
    for i in $(seq 1 "$N"); do
      NAME="$PREFIX-$(date -u +%m%d)-$i-$RANDOM"
      gcloud compute instances create "$NAME" --zone "$ZONE" \
        --machine-type "$MACHINE" --image "$IMAGE" \
        --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
        --provisioning-model SPOT --instance-termination-action DELETE \
        --scopes storage-rw,compute-rw \
        --labels role=solar-worker \
        --metadata startup-script="$STARTUP" --quiet \
        --format="value(name,status)" || echo "  could not create worker $i (quota?)"
    done
    ;;
  status)
    cd "$(dirname "$0")/.." && .venv/bin/python src/gcs_queue.py
    echo "workers:"; gcloud compute instances list --filter="labels.role=solar-worker" \
      --format="table(name,zone.basename(),machineType.basename(),status,creationTimestamp.date('%H:%M'))"
    ;;
  list)
    gcloud compute instances list --filter="labels.role=solar-worker"
    ;;
  down)
    NAMES=$(gcloud compute instances list --filter="labels.role=solar-worker" --format="value(name,zone.basename())")
    [ -n "$NAMES" ] || { echo "no workers"; exit 0; }
    echo "$NAMES" | while read -r name zone; do
      gcloud compute instances delete "$name" --zone "$zone" --quiet &
    done; wait
    ;;
  *)
    sed -n '2,12p' "$0"; exit 2 ;;
esac
