# Sypra

Sypra lets you run AI models on your own computer. You can chat with a local
model, search your documents, create a RAG knowledge base, and fine-tune a
small model, all without sending your data to a cloud service by default.

Your chats, documents, embeddings, and local models stay on your machine.
Sypra uses the internet only when you turn on web search or add a cloud model.

## Install Sypra (regular users)

Use this option if you want to use Sypra and do not plan to change its code.
Package managers install a versioned release, create the `sypra` command, and
handle upgrades and removal without asking you to run a downloaded script.

Windows PowerShell:

```powershell
winget install --id SypraLabs.Sypra --exact
sypra start
```

macOS Apple Silicon or Linux x64:

```sh
brew install --cask sypra-labs/tap/sypra
sypra start
```

For a managed Debian/RPM package, use your organization's configured APT or
DNF repository. Until the public package listings are published, download a
versioned archive from [GitHub Releases](https://github.com/sypra-labs/sypra-ai/releases),
verify its SHA-256 checksum, extract it, and run `sypra start` from the extracted folder.

`sypra start` opens the app in your browser. Add a downloaded model from
**Models**, then choose **Run** to use it. The installer does not download AI
models automatically because models can be large and you choose which ones to
use.

## Run from source (contributors)

Use this option if you are developing Sypra, contributing code, or running a
copy directly from this repository. These instructions are for Windows. You
need [Go 1.23 or newer](https://go.dev/dl/), [Node.js 20 or newer](https://nodejs.org/), and pnpm.

1. Open PowerShell in the Sypra project folder.

2. Start the Sypra API. Leave this terminal open.

   ```powershell
   $env:GOCACHE = "$PWD\.cache\go-build"
   $env:GOMODCACHE = "$PWD\.cache\go-mod"
   go run .\cmd\sypra serve
   ```

3. Open a second PowerShell window in the same project folder. Start the web app.

   ```powershell
   Set-Location .\web
   pnpm install
   pnpm run dev
   ```

4. Open the address shown by pnpm, normally `http://127.0.0.1:5173`, in your browser.

5. Open **Models** in Sypra. Add the folder containing your downloaded model,
   then select **Run**. A valid Qwen model folder contains files such as
   `config.json`, `tokenizer.json`, and one or more `.safetensors` files.

6. Select **New chat**, choose the running model, and send a message.

The model runner is built into Sypra: use the **Run** and **Stop** controls in
**Models**. You do not need to start a separate runner for the web app.

### If you do not have a model yet

You can still explore the interface with the optional mock model. In a third
terminal, run:

```powershell
go run .\cmd\sypra-mock --listen 127.0.0.1:8082
```

Then add `http://127.0.0.1:8082/v1` in **Models** as an external local model.
The mock model is only for trying the app; it does not produce real AI answers.

## Everyday use

### Chat with a model

Start a model from **Models**, then open **New chat**. The first base model you
add becomes the default. You can choose another model before sending a message.

### Ask questions about your files (RAG)

Open **RAGs**, create a collection, and add files, a folder, pasted text, or a
web page. When indexing finishes, attach that RAG to a new chat. Sypra finds
relevant passages from your files and gives them to the model with each
question. You can turn retrieval on or off for individual messages.

### Fine-tune a model

Open **Fine-tuning**, choose a local base model, upload training data, choose
LoRA or QLoRA, and start a run. When it is complete, the result appears under
**Fine-tuning** and can be selected for a new chat. Fine-tuning can take a long
time and benefits from a GPU, but Sypra can also use the CPU.

### Keep your work safe

Use the backup command before moving computers or making major changes. Your
downloaded model files are not included in a workspace backup, so keep those
files separately.

## Optional: use Sypra from the terminal

Most people can use the web app. These commands are helpful if you prefer a
terminal or need to troubleshoot.

```powershell
# See the models known to Sypra
go run .\cmd\sypra models list

# Start a folder-bound terminal chat. Keep the printed session ID to continue it.
go run .\cmd\sypra chat --prompt "Summarize the files in this project."
go run .\cmd\sypra chat --session SESSION_ID --prompt "Now give me the next step."
go run .\cmd\sypra chat --list
go run .\cmd\sypra chat --session SESSION_ID --close

# Create and query a document collection
go run .\cmd\sypra rag create --name handbook --file .\examples\handbook.txt
go run .\cmd\sypra rag query --collection COLLECTION_ID --query "expense policy"

# Start a fine-tuning job and see its progress
go run .\cmd\sypra train start --name support --model C:\models\Qwen3-0.6B --data C:\data\train.jsonl --method lora
go run .\cmd\sypra train list

# Save a backup
go run .\cmd\sypra backup --output .\backups\workspace.sypra-backup
```

To run a model directly from the terminal instead of through the app, use:

```powershell
go run .\cmd\sypra runner --model C:\models\Qwen3-0.6B --listen 127.0.0.1:8081
```

This starts the same local Sypra runner used by the app. See
its OpenAI-compatible endpoint at `http://127.0.0.1:8081/v1` for terminal
requests.

Terminal chats are normal Sypra chats and appear in the web sidebar. Each one
is permanently bound to the folder where it started. While its CLI session is
active, the web app shows its history but cannot send messages to it. Close the
session from that original folder to continue the chat in the web app.

## Where Sypra stores things

Sypra stores its workspace in the `data` folder. The default database is
`data/sypra.db`. Model files remain wherever you downloaded them. Backups
include chats, RAG data, and settings, but not the original model files.

Copy `.env.example` to `.env` only when you want to configure optional cloud
providers, web search, or advanced settings. Never share or commit `.env`,
model files, workspace data, or API keys. Keep the local API bound to a trusted
network unless you add your own access controls.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the small set of local checks to run
before sharing a change.

Sypra is licensed under [Apache-2.0](LICENSE).
