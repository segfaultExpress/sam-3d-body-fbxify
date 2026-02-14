# Postman – Fbxify Worker API

Use these files to test the fbxify worker API against any base URL (e.g. `http://localhost:8000` or your vast.ai worker URL).

## Files

- **fbxify-worker-api.postman_collection.json** – Collection of requests (health, pose, FBX, detection, tracking, job status, download, cancel).
- **fbxify-worker.postman_environment.json** – Environment with variables: `base_url`, `api_key`, `job_id`, `filename`.

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

1. **Health** – GET `{{base_url}}/health` to confirm the worker is up.
2. **Create Pose Job** – POST with `input_file` (image/video). Copy the returned `job_id` into the environment variable `job_id`.
3. **Get Job Status** – GET `{{base_url}}/jobs/{{job_id}}` until `status` is `completed`.
4. **Download Job File** – Set `filename` to one of `output_files` (e.g. `pose_outputs_abc12345.json`) and GET `{{base_url}}/jobs/{{job_id}}/files/{{filename}}`.
5. **Create FBX Job** – POST with `pose_json_file` set to a pose JSON (e.g. from step 4). Then poll status and download the `.fbx` the same way.

For **Create Detection Job** and **Rerun Tracking**, use the same pattern: create job → poll status → download files as needed.
