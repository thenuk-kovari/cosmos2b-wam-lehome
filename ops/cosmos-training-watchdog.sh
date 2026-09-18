#!/usr/bin/env bash

set -uo pipefail

: "${LAMBDA_API_KEY:?LAMBDA_API_KEY is required}"
: "${TRAINING_CONTAINER:?TRAINING_CONTAINER is required}"
: "${CHECKPOINT_S3_PREFIX:?CHECKPOINT_S3_PREFIX is required}"

TARGET_STEP="${TARGET_STEP:-8000}"
INACTIVITY_SECONDS="${INACTIVITY_SECONDS:-1800}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STATE_DIR="${STATE_DIR:-/var/lib/cosmos-training-watchdog}"
LAMBDA_API_BASE="${LAMBDA_API_BASE:-https://cloud.lambda.ai/api/v1}"
EXPECTED_PUBLIC_IP="${EXPECTED_PUBLIC_IP:-}"
LAMBDA_INSTANCE_ID="${LAMBDA_INSTANCE_ID:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR"

LAST_STEP_FILE="$STATE_DIR/last_step"
LAST_PROGRESS_FILE="$STATE_DIR/last_progress_epoch"
TERMINATION_REASON_FILE="$STATE_DIR/termination-reason.json"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

resolve_instance_id() {
    local response matches match_count
    if [[ -n "$LAMBDA_INSTANCE_ID" ]]; then
        return 0
    fi
    if [[ -z "$EXPECTED_PUBLIC_IP" ]]; then
        log "LAMBDA_INSTANCE_ID or EXPECTED_PUBLIC_IP is required"
        return 1
    fi

    response="$(curl --fail --silent --show-error \
        --connect-timeout 10 --max-time 30 \
        --header 'accept: application/json' \
        --header "Authorization: Bearer ${LAMBDA_API_KEY}" \
        "${LAMBDA_API_BASE}/instances")" || return 1
    matches="$(jq -c --arg ip "$EXPECTED_PUBLIC_IP" '[.data[] | select(.ip == $ip)]' <<<"$response")"
    match_count="$(jq 'length' <<<"$matches")"
    if [[ "$match_count" != "1" ]]; then
        log "expected exactly one Lambda instance with IP $EXPECTED_PUBLIC_IP; found $match_count"
        return 1
    fi
    LAMBDA_INSTANCE_ID="$(jq -r '.[0].id' <<<"$matches")"
    log "resolved Lambda instance $LAMBDA_INSTANCE_ID for IP $EXPECTED_PUBLIC_IP"
}

write_state() {
    local step="$1"
    local progress_epoch="$2"
    printf '%s\n' "$step" >"$LAST_STEP_FILE"
    printf '%s\n' "$progress_epoch" >"$LAST_PROGRESS_FILE"
}

checkpoint_step() {
    local marker
    marker="$(aws s3 cp "${CHECKPOINT_S3_PREFIX%/}/checkpoints/latest_checkpoint.txt" - 2>/dev/null)" || return 1
    if [[ "$marker" =~ iter_0*([0-9]+) ]]; then
        printf '%s\n' "$((10#${BASH_REMATCH[1]}))"
        return 0
    fi
    return 1
}

verify_instance_target() {
    local response instance_ip status terminate_available
    response="$(curl --fail --silent --show-error \
        --connect-timeout 10 --max-time 30 \
        --header 'accept: application/json' \
        --header "Authorization: Bearer ${LAMBDA_API_KEY}" \
        "${LAMBDA_API_BASE}/instances/${LAMBDA_INSTANCE_ID}")" || return 1

    instance_ip="$(jq -r '.data.ip // empty' <<<"$response")"
    status="$(jq -r '.data.status // empty' <<<"$response")"
    terminate_available="$(jq -r '.data.actions.terminate.available // false' <<<"$response")"

    if [[ -n "$EXPECTED_PUBLIC_IP" && "$instance_ip" != "$EXPECTED_PUBLIC_IP" ]]; then
        log "refusing termination: instance $LAMBDA_INSTANCE_ID has IP $instance_ip, expected $EXPECTED_PUBLIC_IP"
        return 1
    fi
    if [[ "$status" != "active" && "$status" != "unhealthy" ]]; then
        log "instance already has non-running provider status: $status"
        return 2
    fi
    if [[ "$terminate_available" != "true" ]]; then
        log "termination is not currently available for instance $LAMBDA_INSTANCE_ID"
        return 1
    fi
    return 0
}

record_termination_reason() {
    local reason="$1"
    local observed_step="$2"
    local checkpoint="$3"
    local timestamp
    timestamp="$(date --iso-8601=seconds)"
    jq -n \
        --arg reason "$reason" \
        --arg timestamp "$timestamp" \
        --arg instance_id "$LAMBDA_INSTANCE_ID" \
        --arg container "$TRAINING_CONTAINER" \
        --argjson observed_step "$observed_step" \
        --argjson checkpoint_step "$checkpoint" \
        --argjson target_step "$TARGET_STEP" \
        '{reason: $reason, timestamp: $timestamp, instance_id: $instance_id,
          container: $container, observed_step: $observed_step,
          checkpoint_step: $checkpoint_step, target_step: $target_step}' \
        >"$TERMINATION_REASON_FILE"

    aws s3 cp "$TERMINATION_REASON_FILE" \
        "${CHECKPOINT_S3_PREFIX%/}/watchdog/termination-reason.json" >/dev/null
}

request_termination() {
    local reason="$1"
    local observed_step="$2"
    local checkpoint="$3"
    local response_file http_code

    log "termination condition met: reason=$reason observed_step=$observed_step checkpoint_step=$checkpoint"
    if ! record_termination_reason "$reason" "$observed_step" "$checkpoint"; then
        log "could not persist termination reason to S3; will retry"
        return 1
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN=1: termination request suppressed"
        return 2
    fi
    verify_instance_target
    case "$?" in
        0) ;;
        2) return 2 ;;
        *) return 1 ;;
    esac

    response_file="$(mktemp "$STATE_DIR/lambda-response.XXXXXX")"
    http_code="$(curl --silent --show-error \
        --connect-timeout 10 --max-time 30 \
        --output "$response_file" --write-out '%{http_code}' \
        --request POST \
        --header 'accept: application/json' \
        --header 'content-type: application/json' \
        --header "Authorization: Bearer ${LAMBDA_API_KEY}" \
        --data "{\"instance_ids\":[\"${LAMBDA_INSTANCE_ID}\"]}" \
        "${LAMBDA_API_BASE}/instance-operations/terminate")" || {
            log "Lambda termination request failed before an HTTP response; will retry"
            rm -f "$response_file"
            return 1
        }

    if [[ "$http_code" =~ ^2 ]]; then
        log "Lambda accepted provider termination (HTTP $http_code); billing will end when termination completes"
        cp "$response_file" "$STATE_DIR/lambda-termination-response.json"
        rm -f "$response_file"
        return 0
    fi

    log "Lambda termination request returned HTTP $http_code; will retry"
    rm -f "$response_file"
    return 1
}

now="$(date +%s)"
last_step=0
last_progress="$now"
if [[ -s "$LAST_STEP_FILE" ]]; then
    read -r last_step <"$LAST_STEP_FILE"
fi
if [[ -s "$LAST_PROGRESS_FILE" ]]; then
    read -r last_progress <"$LAST_PROGRESS_FILE"
fi
write_state "$last_step" "$last_progress"

until resolve_instance_id; do
    log "could not resolve the Lambda instance; retrying"
    sleep "$POLL_SECONDS"
done

log "watching container=$TRAINING_CONTAINER target_step=$TARGET_STEP inactivity_seconds=$INACTIVITY_SECONDS"

while true; do
    now="$(date +%s)"
    container_status="$(docker inspect --format '{{.State.Status}}' "$TRAINING_CONTAINER" 2>/dev/null || true)"
    checkpoint="$(checkpoint_step || printf '0')"

    if [[ -z "$container_status" ]]; then
        request_termination "training_container_missing" "$last_step" "$checkpoint"
        result=$?
        [[ "$result" -eq 2 ]] && exit 0
        sleep "$POLL_SECONDS"
        continue
    fi
    if [[ "$container_status" != "running" ]]; then
        request_termination "training_container_${container_status}" "$last_step" "$checkpoint"
        result=$?
        [[ "$result" -eq 2 ]] && exit 0
        sleep "$POLL_SECONDS"
        continue
    fi

    latest_step="$(docker logs --tail 4000 "$TRAINING_CONTAINER" 2>&1 \
        | sed -nE 's/.*] ([0-9]+) : iter_speed .*/\1/p' \
        | tail -n 1)"
    latest_step="${latest_step:-0}"
    if (( latest_step > last_step )); then
        last_step="$latest_step"
        last_progress="$now"
        write_state "$last_step" "$last_progress"
        log "progress observed: step=$last_step checkpoint_step=$checkpoint"
    fi

    if (( checkpoint >= TARGET_STEP )); then
        request_termination "target_checkpoint_complete" "$last_step" "$checkpoint"
        result=$?
        [[ "$result" -eq 2 ]] && exit 0
        sleep "$POLL_SECONDS"
        continue
    fi

    inactive_for=$((now - last_progress))
    if (( inactive_for >= INACTIVITY_SECONDS )); then
        request_termination "no_step_progress_${inactive_for}_seconds" "$last_step" "$checkpoint"
        result=$?
        [[ "$result" -eq 2 ]] && exit 0
        sleep "$POLL_SECONDS"
        continue
    fi

    sleep "$POLL_SECONDS"
done
