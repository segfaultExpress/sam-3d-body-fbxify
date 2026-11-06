# Postman – Fbxify Worker API

Use these files to test the fbxify worker API against any base URL (e.g. `http://localhost:8000` or your vast.ai worker URL).

## Files

- **fbxify-worker-api.postman_collection.json** – Collection of requests (health, models, mounts, pose, FBX, detection, tracking, job status, download, cancel, cleanup, storage, reload).
- **fbxify-worker.postman_environment.json** – Environment with variables: `base_url`, `api_key`, `job_id`, `filename`, `mount_id`.

## Setup

1. **Import in Postman**
   - File → Import → drag or select both the collection and the environment.

2. **Set base URL**
   - Select the environment (e.g. "Fbxify Worker (local)").
   - Edit the environment and set `base_url` to your worker URL, e.g.:
     - Local: `http://localhost:8000`
     - Remote: `http://<host>:8000` or `https://...`

3. **Optional: API key**
   - If the worker uses `FBXIFY_SHARED_SECRET`, set `api_key` in the environment.
   - In each request that needs auth, enable the **Authorization** or **X-API-Key** header (they are present but disabled by default).

## Usage

1. **Health** – GET `{{base_url}}/health`. `200` with `"ready": true` means mesh assets and models are loaded. `503` means the worker is up but must not receive pose jobs. `GET /live` only checks that the process is answering.
2. **Create Pose Job** – POST with `input_file` (image/video). Copy the returned `job_id` into the environment variable `job_id`. Optional form field `fast_inference=true` skips the second body-decoder pass (Fast SAM 3D Body); `body_only=true` skips the hand decoders altogether and batches across frames with `frame_batch_size`.
3. **Get Job Status** – GET `{{base_url}}/jobs/{{job_id}}` until `status` is `completed`.
4. **Download Job File** – Set `filename` to one of `output_files` (e.g. `pose_outputs_abc12345.json`) and GET `{{base_url}}/jobs/{{job_id}}/files/{{filename}}`.
5. **Create FBX Job** – POST with `pose_json_file` set to a pose JSON (e.g. from step 4). Then poll status and download the `.fbx` the same way.

For **Create Detection Job** and **Rerun Tracking**, use the same pattern: create job → poll status → download files as needed.

## Mounts (persistent file storage)

For iterative workflows where you run the same large file through multiple jobs (e.g. re-running tracking dozens of times with different parameters), use **mounts** to upload the file once and reference it by ID:

1. **Mount File** – Open the **Mounts** folder in the collection. Use **Mount File** to upload your estimation JSON (or video). The `mount_id` is automatically saved to `{{mount_id}}`.
2. **List Mounts** – Use **List Mounts** to see all currently mounted files.
3. **Use in jobs** – Use the `(mounted)` variants of job requests (e.g. "Rerun Tracking (mounted)", "Create FBX Job (mounted)"). These reference `{{mount_id}}` instead of uploading a file. You can also enable the `*_mount_id` field (disabled by default) on the standard job requests.
4. **Delete Mount** – Use **Delete Mount** to remove a single mount, or run **Cleanup Temp** with `unmount_all=true` to remove all mounts at once.

Mounts persist across cleanup runs (unless `unmount_all=true` is set) and survive worker restarts. They are stored outside the temp directory.

## Models (catalogue)

The worker keeps one bbox detector, one image encoder and one SAM 3D Body checkpoint resident, chosen from `fbxify/models.json` plus your overlay (see the "Model catalogue" section of the top-level README). The **Models** folder covers it:

1. **List Models** – GET `{{base_url}}/models`. Shows every enabled entry with its fetch `status`, plus `loaded` (what is resident) and `defaults`. `GET /health` also reports `models`.
2. **Select Models** – POST `body_model` / `encoder` / `detector` catalogue names to swap now; the previous weights are freed and the CUDA cache flushed first. `409` while a job is running.
3. **Fetch Models** – POST to download any enabled entries that are still missing (after editing the overlay). Local-mount entries are only checked.

The same `body_model` / `encoder` / `detector` fields are present (disabled by default) on **Create Pose Job** and `detector` on **Create Detection Job**; enabling them swaps for that job. Only catalogue names are accepted; paths, disabled entries and incompatible combinations (e.g. an INT8 encoder with a body model it was not exported for) return `400`.
