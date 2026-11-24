import React, { FormEvent, KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = 'http://localhost:8080/api/v1/';
const API_ORIGIN = new URL(API).origin;

function uiLog(level: 'info' | 'error', event: string, details: Record<string, unknown>) {
  const entry = { event, at: new Date().toISOString(), ...details };
  console[level](`[Sypra UI] ${event}`, entry);

  if (level === 'error') {
    void fetch(API + 'client-logs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level, ...entry }),
      keepalive: true,
    }).catch(() => undefined);
  }
}

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const isForm = options.body instanceof FormData;
  const method = options.method || 'GET';
  const url = API + path;
  const started = performance.now();
  uiLog('info', 'API request started', { method, url });

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers: isForm ? options.headers : { 'Content-Type': 'application/json', ...options.headers },
    });
  } catch (error) {
    uiLog('error', 'API network failure', {
      method,
      url,
      durationMs: Math.round(performance.now() - started),
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }

  const raw = await response.text();
  let body: unknown = raw;
  try {
    body = raw ? JSON.parse(raw) : null;
  } catch {
    // Preserve a non-JSON proxy/server error verbatim.
  }

  if (!response.ok) {
    const message = typeof body === 'object' && body && 'error' in body && typeof body.error === 'string'
      ? body.error
      : raw || `Request failed (${response.status})`;
    uiLog('error', 'API request failed', {
      method,
      url,
      status: response.status,
      statusText: response.statusText,
      durationMs: Math.round(performance.now() - started),
      error: message,
    });
    throw new Error(message);
  }

  uiLog('info', 'API request completed', {
    method,
    url,
    status: response.status,
    durationMs: Math.round(performance.now() - started),
  });
  return body as T;
}

type Model = {
  id: string;
  name: string;
  path: string;
  format: string;
  endpoint: string;
  provider?: 'openai' | 'xai' | 'gemini';
  remoteId?: string;
	family?: string;
	modalities?: string[];
  kind?: 'base' | 'slm';
  default?: boolean;
	parentModel?: string;
  method?: string;
	version?: string;
	status?: 'candidate' | 'production' | 'archived';
  metrics?: Record<string, number>;
};
type RunnerStatus = {
  modelId: string;
  baseModelId?: string;
  runtime?: string;
  state: 'stopped' | 'starting' | 'running' | 'failed' | 'unsupported';
  endpoint?: string;
  pid?: number;
  error?: string;
};
type LocalCatalogEntry = {
  name: string;
  path: string;
  family: string;
  format: string;
  registered: boolean;
};
type Collection = {
  id: string;
  name: string;
  embeddingModel?: string;
  chunkUnit?: string;
  embeddingVersion?: string;
  chunkSize: number;
  documentCount: number;
		 routingTags?: string[];
};
type RAGDocument = {
  id: string;
  collectionId: string;
  name: string;
  source: string;
  path?: string;
  url?: string;
  contentType?: string;
  metadata?: Record<string, string>;
  extractionStatus: 'complete' | 'pending' | 'failed' | string;
  extractionError?: string;
  extractor?: string;
  chunkCount: number;
  refreshIntervalMinutes?: number;
  lastRefreshedAt?: string;
  nextRefreshAt?: string;
};
type RetrievalResult = {
  chunk: { id: string; documentId?: string; source: string; content: string; position: number; metadata?: Record<string, string> };
  score: number;
  semanticScore: number;
  lexicalScore: number;
  previous?: { content: string; position: number };
  next?: { content: string; position: number };
  document?: RAGDocument;
  location: string;
};
type EvaluationSet = { id: string; collectionId: string; name: string; cases: { id: string; query: string; relevantDocumentIds?: string[]; relevantChunkIds?: string[] }[] };
type EvaluationRun = { id: string; metrics: Record<string, number>; createdAt: string };
type Project = {
  id: string;
  name: string;
};
type Chat = {
  id: string;
  title: string;
  projectId?: string;
  modelId?: string;
  collectionId?: string;
  ragMode?: 'focused' | 'full';
  ragTopK?: number;
  ragEnabled?: boolean;
  webEnabled?: boolean;
  pinned?: boolean;
  workspacePath?: string;
  cliSessionActive?: boolean;
};
type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
	parts?: ContentPart[];
  sources?: string[];
  status?: 'working' | 'completed' | 'failed' | 'cancelled';
	createdAt?: string;
	latencyMs?: number;
  pending?: boolean;
};
type ChatStreamEvent = {
  messageID: string;
  content: string;
	parts?: ContentPart[];
  status: 'working' | 'completed' | 'failed' | 'cancelled';
  sources?: string[];
	latencyMs?: number;
};
type ContentPart = { type: 'text' | 'markdown' | 'diagram' | 'image_url' | 'image' | 'video_url' | 'video' | 'input_audio' | 'audio' | 'file' | 'artifact'; text?: string; url?: string; data?: string; name?: string; mimeType?: string };
type MediaJob = { id: string; kind: 'ocr' | 'transcription' | 'image_generation' | 'image_edit'; prompt?: string; inputName?: string; status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'; phase?: string; progress: number; outputUrl?: string; outputMime?: string; text?: string; error?: string; createdAt: string };
type ToolDefinition = { id: string; name: string; description: string; source: string; inputSchema: Record<string, unknown> };
type ToolConnection = { id: string; name: string; kind: 'mcp' | 'pipeline'; command?: string; args?: string[]; toolIds?: string[]; enabled: boolean; status?: string; error?: string };
type ToolPermission = { id: string; toolId: string; modelId?: string; projectId?: string; enabled: boolean; requireApproval: boolean };
type AgentRun = { id: string; modelId?: string; projectId?: string; toolId: string; arguments: Record<string, unknown>; status: string; output?: string; error?: string; createdAt: string };
type Job = {
  id: string;
  name: string;
  method: string;
  status: string;
  progress: number;
	phase?: string;
	epoch?: number;
	epochs?: number;
	step?: number;
	steps?: number;
	tokens?: number;
	tokensPerSecond?: number;
  trainLoss?: number;
  evalLoss?: number;
	error?: string;
};
type Metrics = {
  requests: number;
  trainingJobs: number;
};
type Overlay =
  | { kind: 'delete'; target: 'chat' | 'rag' | 'model'; id: string }
  | { kind: 'rename'; chat: Chat }
  | { kind: 'project'; chat?: Chat }
  | { kind: 'profile' }
  | { kind: 'settings' };

const list = <T,>(value: T[] | null) => Array.isArray(value) ? value : [];

async function attachmentParts(files: File[]): Promise<ContentPart[]> {
  return Promise.all(files.map(file => new Promise<ContentPart>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error || new Error(`Unable to read ${file.name}`));
    reader.onload = () => {
      const data = String(reader.result || '');
      if (file.type.startsWith('image/')) resolve({ type: 'image_url', data, name: file.name, mimeType: file.type });
      else if (file.type.startsWith('audio/')) resolve({ type: 'input_audio', data, name: file.name, mimeType: file.type });
      else if (file.type.startsWith('video/')) resolve({ type: 'video_url', data, name: file.name, mimeType: file.type });
      else resolve({ type: 'file', data, name: file.name, mimeType: file.type });
    };
    reader.readAsDataURL(file);
  })));
}

function PaperclipIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21.4 11.6-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 1 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5" /></svg>;
}

function NewChatIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8" /><path d="m13.5 6.5 4 4M11 17l-4 1 1-4 8.8-8.8a1.4 1.4 0 0 1 2 2L11 17Z" /></svg>;
}

function FolderIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7.5a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-9Z" /></svg>;
}

function GlobeIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" /></svg>;
}

function SearchIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8" /><path d="m16 16 4.2 4.2" /></svg>;
}

function ArrowUpIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 7-7 7 7M12 5v14" /></svg>;
}

function MoreIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="5" cy="12" r="1.7" /><circle cx="12" cy="12" r="1.7" /><circle cx="19" cy="12" r="1.7" /></svg>;
}

function PinIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m14.5 3 6.5 6.5-2.8 1.1-3.4 3.4 1.1 4.1-1.3 1.3-4.1-4.1-4.6 4.6-.9-.9 4.6-4.6-4.1-4.1 1.3-1.3 4.1 1.1 3.4-3.4L14.5 3Z" /></svg>;
}

function CopyIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="11" height="12" rx="2" /><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h2" /></svg>;
}

function DownloadIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v11m0 0 4-4m-4 4-4-4M5 20h14" /></svg>;
}

function RAGIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6.5 12 3l8 3.5-8 3.5-8-3.5ZM4 12l8 3.5 8-3.5M4 17.5 12 21l8-3.5" /></svg>;
}

function ToolsIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.7 6.3a4.2 4.2 0 0 0-5.5 5.5L3.5 17.5a2.1 2.1 0 1 0 3 3l5.7-5.7a4.2 4.2 0 0 0 5.5-5.5l-2.6 2.6-2.1-2.1 2.7-2.5Z" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

type MenuPosition = { top: number; left: number };

function menuPosition(button: HTMLButtonElement, width = 168): MenuPosition {
  const bounds = button.getBoundingClientRect();
  return {
    top: Math.min(window.innerHeight - 12, bounds.bottom + 5),
    left: Math.max(8, Math.min(window.innerWidth - width - 8, bounds.right - width)),
  };
}

function SettingsIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3" /><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.09a2 2 0 0 1 1 1.74v.5a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.09a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2Z" /></svg>;
}

function App() {
  const [tab, setTab] = useState<'chat' | 'rag' | 'training' | 'models' | 'media' | 'tools'>('chat');
  const [models, setModels] = useState<Model[]>([]);
  const [runners, setRunners] = useState<RunnerStatus[]>([]);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [chats, setChats] = useState<Chat[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
	const [mediaJobs, setMediaJobs] = useState<MediaJob[]>([]);
	const [toolCatalog, setToolCatalog] = useState<ToolDefinition[]>([]);
	const [toolConnections, setToolConnections] = useState<ToolConnection[]>([]);
	const [toolPermissions, setToolPermissions] = useState<ToolPermission[]>([]);
	const [agentRuns, setAgentRuns] = useState<AgentRun[]>([]);
  const [metrics, setMetrics] = useState<Metrics>();
  const [active, setActive] = useState<Chat>();
  const [draft, setDraft] = useState<Partial<Pick<Chat, 'projectId' | 'modelId' | 'collectionId' | 'ragMode' | 'ragTopK' | 'ragEnabled' | 'webEnabled'>>>({});
  const [draftVersion, setDraftVersion] = useState(0);
  const [histories, setHistories] = useState<Record<string, Message[]>>({});
  const [filter, setFilter] = useState('');
  const [chatSearchOpen, setChatSearchOpen] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [ragEditor, setRagEditor] = useState<Collection | 'new'>();
	const [trainingSetupOpen, setTrainingSetupOpen] = useState(false);
	const [navigationOpen, setNavigationOpen] = useState(false);
  const [overlay, setOverlay] = useState<Overlay>();
  const [apiOnline, setAPIOnline] = useState(false);

  useEffect(() => {
    let icon = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
    if (!icon) {
    icon = document.createElement('link');
    icon.rel = 'icon';
    document.head.append(icon);
    }
    icon.href = '/sypra.ico';
  }, []);

  useEffect(() => {
    if (!notice && !error)
      return;
    const timeout = window.setTimeout(() => {
      setNotice('');
      setError('');
    }, 1000);
    return () => window.clearTimeout(timeout);
  }, [notice, error]);

  const refresh = async () => {
    try {
      const [modelsResponse, runnersResponse, collectionsResponse, projectsResponse, chatsResponse, jobsResponse, mediaJobsResponse, toolsResponse, connectionsResponse, permissionsResponse, agentRunsResponse, metricsResponse] = await Promise.all([
        api<Model[] | null>('models'),
        api<RunnerStatus[] | null>('runners'),
        api<Collection[] | null>('rag/collections'),
        api<Project[] | null>('projects'),
        api<Chat[] | null>('chats'),
        api<Job[] | null>('training/jobs'),
			api<MediaJob[] | null>('media/jobs'),
			api<ToolDefinition[] | null>('tools/catalog'),
			api<ToolConnection[] | null>('tools/connections'),
			api<ToolPermission[] | null>('tools/permissions'),
			api<AgentRun[] | null>('agents/runs'),
        api<Metrics>('metrics'),
      ]);
      const nextChats = list(chatsResponse);
      setModels(list(modelsResponse));
      setRunners(list(runnersResponse));
      setCollections(list(collectionsResponse));
      setProjects(list(projectsResponse));
      setChats(nextChats);
      setJobs(list(jobsResponse));
			setMediaJobs(list(mediaJobsResponse));
			setToolCatalog(list(toolsResponse));
			setToolConnections(list(connectionsResponse));
			setToolPermissions(list(permissionsResponse));
			setAgentRuns(list(agentRunsResponse));
      setMetrics(metricsResponse);
      setAPIOnline(true);
      setActive(current => current ? nextChats.find(chat => chat.id === current.id) || current : current);
    } catch (e) {
      setAPIOnline(false);
      setError((e as Error).message);
    }
  };

  const checkAPIHealth = async () => {
    try {
      const response = await fetch(`${API_ORIGIN}/healthz`, { cache: 'no-store' });
      setAPIOnline(response.ok);
    } catch {
      setAPIOnline(false);
    }
  };

  const loadHistory = async (chatID: string) => {
    const serverHistory = list(await api<Message[] | null>(`chats/${chatID}/messages`));
    setHistories(current => {
      const localPending = (current[chatID] || []).filter(message => message.pending);
      const pendingUser = localPending.find(message => message.role === 'user');
      const requestIsPersisted = Boolean(
        pendingUser && serverHistory.some(message => message.role === 'user' && message.content === pendingUser.content),
      );
      const pending = requestIsPersisted || serverHistory.some(message => message.status === 'working')
        ? []
        : localPending;
      return { ...current, [chatID]: [...serverHistory, ...pending] };
    });
  };

  const open = async (chat: Chat, updateURL = true) => {
    setActive(chat);
    setDraft({});
    setTab('chat');
    if (updateURL && window.location.pathname !== `/chat/${chat.id}`) {
      window.history.pushState({}, '', `/chat/${chat.id}`);
    }
    try {
      await loadHistory(chat.id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  useEffect(() => {
    void refresh();
    void checkAPIHealth();
    const refreshTimer = window.setInterval(refresh, 5000);
    const healthTimer = window.setInterval(checkAPIHealth, 5000);
    return () => {
      window.clearInterval(refreshTimer);
      window.clearInterval(healthTimer);
    };
  }, []);

  useEffect(() => {
    const followURL = () => {
      const match = /^\/chat\/([0-9a-f-]{36})$/i.exec(window.location.pathname);
      const chat = match && chats.find(item => item.id === match[1]);
      if (chat && chat.id !== active?.id) {
        void open(chat, false);
      }
    };
    followURL();
    window.addEventListener('popstate', followURL);
    return () => window.removeEventListener('popstate', followURL);
  }, [chats, active?.id]);

  const create = (projectId = '', initial: Partial<Chat> = {}) => {
    setActive(undefined);
    setDraft({ ...initial, projectId });
    setDraftVersion(current => current + 1);
    setTab('chat');
    window.history.pushState({}, '', '/');
  };
  const bind = async (kind: 'modelId' | 'collectionId' | 'ragMode' | 'ragTopK' | 'ragEnabled' | 'webEnabled', value: string | number | boolean) => {
    try {
      if (!active) {
        setDraft(current => ({ ...current, [kind]: value }));
        return;
      }
      const bindingChange = kind === 'collectionId';
      if (bindingChange && active.collectionId && value !== active.collectionId) {
        setError('This chat is permanently bound to its RAG collection. Start a new chat to use different knowledge.');
        return;
      }
      const chat = await api<Chat>(`chats/${active.id}`, { method: 'PATCH', body: JSON.stringify({ [kind]: value }) });
      setActive(chat);
      setChats(current => current.map(item => item.id === chat.id ? chat : item));
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const append = (chatID: string, messages: Message[]) => {
    const createdAt = new Date().toISOString();
    setHistories(current => ({ ...current, [chatID]: [...(current[chatID] || []), ...messages.map(message => ({ ...message, createdAt: message.createdAt || createdAt }))] }));
  };
  const settle = (chatID: string, response: string, sources?: string[]) => {
    setHistories(current => ({ ...current, [chatID]: (current[chatID] || []).map(message => message.pending ? { ...message, pending: false, content: message.role === 'assistant' ? response : message.content, sources: message.role === 'assistant' ? sources : message.sources } : message) }));
    void loadHistory(chatID).catch(() => undefined);
  };
  const applyChatStream = useCallback((chatID: string, event: ChatStreamEvent) => {
    setHistories(current => {
      const history = current[chatID] || [];
      let applied = false;
      const next = history.map(message => {
        const isPendingAssistant = !applied && message.role === 'assistant' && (message.pending || message.status === 'working');
        if (message.id !== event.messageID && !isPendingAssistant)
          return message;
        applied = true;
		return { ...message, id: event.messageID, content: event.content, parts: event.parts || message.parts, status: event.status, pending: false, sources: event.sources || message.sources, latencyMs: event.latencyMs || message.latencyMs };
      });
      return { ...current, [chatID]: next };
    });
  }, []);
  const done = (message: string) => { setNotice(message); setError(''); void refresh(); };
  const remove = (kind: 'chat' | 'rag' | 'model', id: string) => setOverlay({ kind: 'delete', target: kind, id });
  const updateChat = async (chatID: string, patch: Partial<Pick<Chat, 'title' | 'projectId' | 'pinned'>>) => {
    const updated = await api<Chat>(`chats/${chatID}`, { method: 'PATCH', body: JSON.stringify(patch) });
    setChats(current => current.map(item => item.id === updated.id ? updated : item));
    setActive(current => current?.id === updated.id ? updated : current);
    return updated;
  };
  const renameChat = async (chatID: string, title: string) => {
    if (!title.trim())
      return;
    try {
      await updateChat(chatID, { title: title.trim() });
      setOverlay(undefined);
      done('Chat renamed.');
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const togglePinnedChat = async (chat: Chat) => {
    try {
      await updateChat(chat.id, { pinned: !chat.pinned });
      done(chat.pinned ? 'Chat unpinned.' : 'Chat pinned.');
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const moveChatToProject = async (chat: Chat, projectID: string) => {
    try {
      await updateChat(chat.id, { projectId: projectID });
      const project = projects.find(item => item.id === projectID);
      done(project ? `Moved to ${project.name}.` : 'Removed from project.');
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const confirmDelete = async () => {
    if (!overlay || overlay.kind !== 'delete')
      return;
    const { target: kind, id } = overlay;
    setOverlay(undefined);
    try {
      const path = kind === 'chat' ? `chats/${id}` : kind === 'rag' ? `rag/collections/${id}` : `models/${id}`;
      await api(path, { method: 'DELETE' });
      if (kind === 'chat' && active?.id === id) {
        setActive(undefined);
        window.history.pushState({}, '', '/');
      }
      done(`${kind === 'rag' ? 'RAG collection' : kind} deleted.`);
    }
    catch (e) {
      setError((e as Error).message);
    }
  };
  const createProject = async (name: string, chat?: Chat) => {
    setOverlay(undefined);
    if (!name.trim())
      return;
    try {
      const project = await api<Project>('projects', { method: 'POST', body: JSON.stringify({ name: name.trim() }) });
      setProjects(current => [project, ...current]);
      if (chat)
        await moveChatToProject(chat, project.id);
      else
        done(`Created ${project.name}.`);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const createRAGChat = async (collectionId: string) => {
    create('', { collectionId });
  };
  const setDefaultBaseModel = async (modelID: string) => {
    try {
      const updated = await api<Model>(`models/${modelID}/default`, { method: 'POST' });
      setModels(current => current.map(model => model.kind === 'base' ? { ...model, default: model.id === updated.id } : model));
      done(`${updated.name} is now the default base model.`);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const startModel = async (modelID: string) => {
    try {
      await api<RunnerStatus>(`models/${modelID}/start`, { method: 'POST' });
      done('Model runner is starting. It will become selectable when healthy.');
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const stopModel = async (modelID: string) => {
    try {
      await api<RunnerStatus>(`models/${modelID}/stop`, { method: 'POST' });
      done('Model runner stopped.');
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const visibleChats = useMemo(
    () => chats.filter(chat => chat.title.toLowerCase().includes(filter.toLowerCase())).sort((left, right) => {
      if (left.pinned !== right.pinned)
        return left.pinned ? -1 : 1;
      return 0;
    }),
    [chats, filter],
  );
  const ungroupedChats = visibleChats.filter(chat => !chat.projectId);
  const baseModels = models.filter(model => model.kind !== 'slm' && !model.provider);
  const cloudModels = models.filter(model => model.kind !== 'slm' && model.provider);
  const slms = models.filter(model => model.kind === 'slm');
  const activeModelIDs = new Set(runners.filter(runner => runner.state === 'running').map(runner => runner.modelId));
  const selectableModels = models.filter(model => Boolean(model.provider) || Boolean(model.endpoint) || activeModelIDs.has(model.id));
  return <main className="app-shell">
  <aside className={`sidebar${navigationOpen ? ' open' : ''}`}>
    <div className="brand">
      <img className="brand-mark" src="/sypra.ico" alt="Sypra logo" />
      <span>Sypra</span>
      <button type="button" className="sidebar-search" aria-label="Search chats" aria-pressed={chatSearchOpen} title="Search chats" onClick={() => setChatSearchOpen(value => !value)}><SearchIcon /></button>
    </div>
    {tab === 'chat' && <button className="new-chat" onClick={() => void create()}>
      <NewChatIcon /> New chat
    </button>}
    {chatSearchOpen && <label className="chat-filter">
      <span>Search chats</span>
      <input value={filter} onChange={event => setFilter(event.target.value)} placeholder="Search chats" />
    </label>}
    <Group title="Chats">
      {ungroupedChats.map(chat => (
        <ChatSidebarEntry
          key={chat.id}
          selected={active?.id === chat.id}
          chat={chat}
          projects={projects}
          choose={() => void open(chat)}
          remove={() => void remove('chat', chat.id)}
          rename={() => setOverlay({ kind: 'rename', chat })}
          togglePin={() => void togglePinnedChat(chat)}
          move={projectID => void moveChatToProject(chat, projectID)}
          createProject={() => setOverlay({ kind: 'project', chat })}
        />
      ))}
    </Group>
    <Group
      title="Projects"
      action={() => setOverlay({ kind: 'project' })}
    >
      {projects.map(project => <ProjectSidebarEntry key={project.id} project={project} chats={visibleChats.filter(chat => chat.projectId === project.id)} projects={projects} selectedChatID={active?.id} choose={chat => void open(chat)} createChat={() => void create(project.id)} remove={chat => void remove('chat', chat.id)} rename={chat => setOverlay({ kind: 'rename', chat })} togglePin={chat => void togglePinnedChat(chat)} move={(chat, projectID) => void moveChatToProject(chat, projectID)} createProjectForChat={chat => setOverlay({ kind: 'project', chat })} />)}
    </Group>
    <Group
      title="RAGs"
      action={() => {
        setRagEditor('new');
      }}
      active={tab === 'rag'}
    >
      {collections.map(collection => (
        <RAGSidebarEntry
          key={collection.id}
          selected={active?.collectionId === collection.id}
          collection={collection}
          choose={() => {
            void bind('collectionId', collection.id);
            setTab('chat');
          }}
          edit={() => {
            setRagEditor(collection);
          }}
          remove={() => void remove('rag', collection.id)}
        />
      ))}
    </Group>
    <Group title="Base models" action={() => setTab('models')} onTitleClick={() => setTab('models')}>
      {baseModels.filter(model => selectableModels.some(candidate => candidate.id === model.id)).map(model => (
        <SidebarEntry
          key={model.id}
          label={model.name}
          plain
          choose={() => { void bind('modelId', model.id); setTab('chat'); }}
          remove={() => void remove('model', model.id)}
        />
      ))}
    </Group>
    <Group title="Cloud models" action={() => setTab('models')} onTitleClick={() => setTab('models')}>
      {cloudModels.map(model => (
        <SidebarEntry
          key={model.id}
          label={model.name}
          choose={() => { void bind('modelId', model.id); setTab('chat'); }}
          remove={() => void remove('model', model.id)}
        />
      ))}
    </Group>
    <Group title="Fine-tuning" action={() => setTrainingSetupOpen(true)} onTitleClick={() => setTab('training')} active={tab === 'training'}>
      {slms.filter(model => selectableModels.some(candidate => candidate.id === model.id)).map(model => <SidebarEntry key={model.id} label={model.name} choose={() => { void bind('modelId', model.id); setTab('chat'); }} remove={() => void remove('model', model.id)} />)}
      {!slms.length && <p className="sidebar-empty">No fine-tuned models yet.</p>}
      <button className="sidebar-section-link" onClick={() => setTab('training')}>Runs & metrics ({jobs.length})</button>
    </Group>
		<Group title="Media workflows" action={() => setTab('media')} active={tab === 'media'}>
			<button onClick={() => setTab('media')}>Jobs & outputs ({mediaJobs.length})</button>
		</Group>
		<Group title="Agents & tools" action={() => setTab('tools')} active={tab === 'tools'}>
			<button onClick={() => setTab('tools')}>Tools & workflows ({toolCatalog.length})</button>
		</Group>
    <footer>
      <div className="sidebar-account">
        <button type="button" className="profile-button" onClick={() => setOverlay({ kind: 'profile' })}>
          <span className="profile-avatar">S</span><span><strong>Local profile</strong><small>Offline workspace</small></span>
        </button>
        <button type="button" className="settings-button" aria-label="Open settings" title="Settings" onClick={() => setOverlay({ kind: 'settings' })}><SettingsIcon /></button>
      </div>
      <span>LOCAL CONTROL PLANE</span><p><strong>{metrics?.requests || 0}</strong> requests · <strong>{metrics?.trainingJobs || 0}</strong> runs</p>
    </footer>
  </aside>
  {navigationOpen && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={() => setNavigationOpen(false)} />}
  <section className="workspace">
    <header><div className="header-title"><button className="menu-toggle" type="button" aria-label="Open navigation" onClick={() => setNavigationOpen(true)}>☰</button>{tab === 'chat' && active && active.title !== 'New chat' && <div><strong>{active.title}</strong>{active.projectId && <small>Project chat</small>}</div>}</div><span className={`local-status ${apiOnline ? 'online' : 'offline'}`} title={apiOnline ? 'API is reachable' : 'API is offline'}><i aria-hidden="true" />{apiOnline ? 'Online' : 'Offline'}</span></header>
    {error && <p className="banner error" role="alert">{error}</p>}{notice && <p className="banner" role="status">{notice}</p>}
    <div className="page-content">
    {tab === 'chat' && <ChatPanel key={active?.id || `new-${draftVersion}`} chat={active} draft={draft} messages={active ? histories[active.id] || [] : []} models={models} activeModelIDs={activeModelIDs} collections={collections} bind={bind} created={open} append={append} settle={settle} reload={loadHistory} onStream={applyChatStream} done={done} fail={setError}/>}
    {tab === 'rag' && <RagPanel collections={collections} chats={chats} openEditor={collection => setRagEditor(collection || 'new')} openChat={open} createRAGChat={createRAGChat} onChanged={refresh} done={done} fail={setError}/>} 
    {tab === 'training' && <TrainingPanel jobs={jobs} baseModels={baseModels} done={done} fail={setError}/>}
		{tab === 'media' && <MediaPanel jobs={mediaJobs} onChanged={refresh} done={done} fail={setError}/>} 
		{tab === 'tools' && <ToolsPanel tools={toolCatalog} connections={toolConnections} permissions={toolPermissions} runs={agentRuns} models={models} projects={projects} onChanged={refresh} done={done} fail={setError}/>} 
    {tab === 'models' && <ModelsPanel models={models} runners={runners} setDefaultBaseModel={setDefaultBaseModel} startModel={startModel} stopModel={stopModel} done={done} fail={setError}/>}
    </div>
  </section>
  <AppOverlay overlay={overlay} onClose={() => setOverlay(undefined)} onDelete={() => void confirmDelete()} onProject={(name, chat) => void createProject(name, chat)} onRename={(chatID, title) => void renameChat(chatID, title)} />
  {ragEditor && <RagEditorOverlay editing={ragEditor === 'new' ? undefined : ragEditor} onClose={() => setRagEditor(undefined)} onCreated={() => setTab('rag')} done={done} fail={setError} />}
  {trainingSetupOpen && <div className="modal-layer" role="presentation" onMouseDown={() => setTrainingSetupOpen(false)}><section className="modal training-setup-modal" role="dialog" aria-modal="true" aria-label="Fine-tune a model" onMouseDown={event => event.stopPropagation()}><div className="modal-heading"><div><p>Fine-tuning</p><h2>Fine-tune a model</h2></div><button type="button" aria-label="Close dialog" onClick={() => setTrainingSetupOpen(false)}>×</button></div><TrainingPanel jobs={jobs} baseModels={baseModels} setupOnly onQueued={() => { setTrainingSetupOpen(false); setTab('training'); }} done={done} fail={setError}/></section></div>}
  </main>;
}
function AppOverlay({ overlay, onClose, onDelete, onProject, onRename }: {
  overlay?: Overlay;
  onClose: () => void;
  onDelete: () => void;
  onProject: (name: string, chat?: Chat) => void;
  onRename: (chatID: string, title: string) => void;
}) {
  const [name, setName] = useState('');
  useEffect(() => { setName(overlay?.kind === 'rename' ? overlay.chat.title : ''); }, [overlay]);
  if (!overlay)
    return null;
  const deleting = overlay.kind === 'delete';
  const title = deleting
    ? `Delete ${overlay.target === 'rag' ? 'RAG collection' : overlay.target}?`
      : overlay.kind === 'rename'
        ? 'Rename chat'
      : overlay.kind === 'profile'
      ? 'Local profile'
      : overlay.kind === 'settings'
        ? 'Settings'
        : overlay.chat ? 'Create project and move chat' : 'Create project';
  const message = deleting
    ? 'This removes it from the workspace. This action cannot be undone.'
    : overlay.kind === 'rename'
      ? 'Choose a clear title that will be easier to find in your chat history.'
    : overlay.kind === 'profile'
      ? 'This workspace runs locally. Chats, RAG collections, models, and training data remain on this machine unless you explicitly enable a cloud integration.'
      : overlay.kind === 'settings'
        ? 'Workspace preferences and local runtime information are managed here.'
        : overlay.chat ? `Create a project and move “${overlay.chat.title}” into it.` : 'Projects keep related chats organized in one workspace.';
  return <div className="modal-layer" role="presentation" onMouseDown={onClose}>
    <section className="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title" onMouseDown={event => event.stopPropagation()}>
      <div className="modal-heading"><div><p>Workspace</p><h2 id="modal-title">{title}</h2></div><button type="button" aria-label="Close dialog" onClick={onClose}>×</button></div>
      <p className="modal-copy">{message}</p>
      {deleting ? <div className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button type="button" className="danger" onClick={onDelete}>Delete</button></div> : overlay.kind === 'project' ? <form onSubmit={event => { event.preventDefault(); onProject(name, overlay.chat); }}><label>Project name<input autoFocus value={name} onChange={event => setName(event.target.value)} placeholder="e.g. Customer support"/></label><div className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button disabled={!name.trim()}>{overlay.chat ? 'Create and move' : 'Create project'}</button></div></form> : overlay.kind === 'rename' ? <form onSubmit={event => { event.preventDefault(); onRename(overlay.chat.id, name); }}><label>Chat title<input autoFocus value={name} onChange={event => setName(event.target.value)} placeholder="Give this chat a name"/></label><div className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button disabled={!name.trim()}>Save</button></div></form> : <div className="modal-actions"><button type="button" onClick={onClose}>Done</button></div>}
    </section>
  </div>;
}
function Group({ title, count, action, onTitleClick, active, children }: {
  title: string;
  count?: number;
  action?: () => void;
  onTitleClick?: () => void;
  active?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={`side-group ${active ? 'active' : ''}`}>
      <div className="side-group-title">
        <button className="side-group-heading" type="button" onClick={onTitleClick} disabled={!onTitleClick}>
          {title}
          {count !== undefined && <em>{count}</em>}
        </button>
        {action && <button type="button" aria-label={`Add ${title}`} title={`Add ${title}`} onClick={action}>+</button>}
      </div>
      {children}
    </section>
  );
}

function SidebarEntry({ label, selected, choose, remove, plain = false }: {
  label: string;
  selected?: boolean;
  choose: () => void;
  remove: () => void;
  plain?: boolean;
}) {
  return (
    <div className={`sidebar-entry ${plain ? 'plain' : ''} ${selected ? 'selected' : ''}`}>
      <button onClick={choose}>{label}</button>
      <button className="sidebar-delete" onClick={remove} aria-label={`Delete ${label}`} title="Delete">
        <MoreIcon />
      </button>
    </div>
  );
}

function ProjectSidebarEntry({ project, chats, projects, selectedChatID, choose, createChat, remove, rename, togglePin, move, createProjectForChat }: {
  project: Project;
  chats: Chat[];
  projects: Project[];
  selectedChatID?: string;
  choose: (chat: Chat) => void;
  createChat: () => void;
  remove: (chat: Chat) => void;
  rename: (chat: Chat) => void;
  togglePin: (chat: Chat) => void;
  move: (chat: Chat, projectID: string) => void;
  createProjectForChat: (chat: Chat) => void;
}) {
  return <section className="project-sidebar-entry">
    <div className="project-sidebar-heading">
      <div className="project-sidebar-toggle"><FolderIcon /><span>{project.name}</span></div>
      <button type="button" className="project-new-chat" onClick={createChat} aria-label={`New chat in ${project.name}`} title={`New chat in ${project.name}`}>+</button>
    </div>
    <div className="project-chat-list">{chats.length ? chats.map(chat => <ChatSidebarEntry key={chat.id} chat={chat} projects={projects} selected={selectedChatID === chat.id} choose={() => choose(chat)} remove={() => remove(chat)} rename={() => rename(chat)} togglePin={() => togglePin(chat)} move={projectID => move(chat, projectID)} createProject={() => createProjectForChat(chat)} />) : <p>No chats yet.</p>}</div>
  </section>;
}

function ChatSidebarEntry({ chat, projects, selected, choose, remove, rename, togglePin, move, createProject }: {
  chat: Chat;
  projects: Project[];
  selected?: boolean;
  choose: () => void;
  remove: () => void;
  rename: () => void;
  togglePin: () => void;
  move: (projectID: string) => void;
  createProject: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [projectsOpen, setProjectsOpen] = useState(false);
  const [position, setPosition] = useState<MenuPosition>();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const close = () => { setOpen(false); setProjectsOpen(false); };

  useEffect(() => {
    if (!open)
      return;
    const closeOutside = (event: MouseEvent) => {
      const target = event.target;
      if (buttonRef.current?.contains(target as Node) || (target instanceof Element && target.closest('[data-sidebar-popover]')))
        return;
      close();
    };
    document.addEventListener('mousedown', closeOutside);
    return () => document.removeEventListener('mousedown', closeOutside);
  }, [open]);

  const toggle = () => {
    if (!buttonRef.current)
      return;
    setPosition(menuPosition(buttonRef.current));
    setOpen(value => !value);
  };

  return (
    <div className={`sidebar-entry chat-sidebar-entry ${selected ? 'selected' : ''}`}>
      <button onClick={choose}>{chat.pinned && <PinIcon />}{chat.title}</button>
      <button ref={buttonRef} className="sidebar-more" onClick={toggle} aria-label={`Chat options for ${chat.title}`} aria-expanded={open} title="Chat options"><MoreIcon /></button>
      {open && position && createPortal(<div className="sidebar-menu sidebar-popover chat-sidebar-menu" data-sidebar-popover="true" style={position}>
        <button onClick={() => { close(); rename(); }}>Rename</button>
        <button onClick={() => { close(); togglePin(); }}>{chat.pinned ? 'Unpin' : 'Pin'}</button>
        <div className="sidebar-submenu-trigger">
          <button onClick={() => setProjectsOpen(value => !value)} aria-expanded={projectsOpen}>Move to project <span>›</span></button>
          {projectsOpen && <div className="sidebar-submenu">
            <button className={!chat.projectId ? 'selected' : ''} onClick={() => { close(); move(''); }}>No project</button>
            {projects.length ? projects.map(project => <button className={chat.projectId === project.id ? 'selected' : ''} key={project.id} onClick={() => { close(); move(project.id); }}>{project.name}</button>) : <p>No projects yet</p>}
            <button onClick={() => { close(); createProject(); }}>+ New project</button>
          </div>}
        </div>
        <button className="danger" onClick={() => { close(); remove(); }}>Delete</button>
      </div>, document.body)}
    </div>
  );
}

function RAGSidebarEntry({ collection, selected, choose, edit, remove }: {
  collection: Collection;
  selected?: boolean;
  choose: () => void;
  edit: () => void;
  remove: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<MenuPosition>();
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open)
      return;
    const closeOutside = (event: MouseEvent) => {
      const target = event.target;
      if (buttonRef.current?.contains(target as Node) || (target instanceof Element && target.closest('[data-sidebar-popover]')))
        return;
      setOpen(false);
    };
    document.addEventListener('mousedown', closeOutside);
    return () => document.removeEventListener('mousedown', closeOutside);
  }, [open]);

  const toggle = () => {
    if (!buttonRef.current)
      return;
    setPosition(menuPosition(buttonRef.current, 120));
    setOpen(value => !value);
  };

  return (
    <div className={`sidebar-entry plain rag-sidebar-entry ${selected ? 'selected' : ''}`}>
      <button onClick={choose}>{collection.name}</button>
      <button
        ref={buttonRef}
        className="sidebar-more"
        onClick={toggle}
        aria-label={`RAG options for ${collection.name}`}
        aria-expanded={open}
      >
        <MoreIcon />
      </button>
      {open && position && createPortal(
        <div className="sidebar-menu sidebar-popover" data-sidebar-popover="true" style={position}>
          <button onClick={() => { setOpen(false); edit(); }}>Edit</button>
          <button className="danger" onClick={() => { setOpen(false); remove(); }}>Delete</button>
        </div>, document.body,
      )}
    </div>
  );
}

function RichText({ value }: { value: string }) {
  const blocks = value.split(/(```[\s\S]*?```)/g).filter(Boolean);
  return <div className="rich-text">{blocks.map((block, index) => {
    const code = /^```([^\n]*)\n?([\s\S]*?)```$/.exec(block);
    if (code) return <pre className="code-block" key={index}><code data-language={code[1] || 'text'}>{code[2]}</code></pre>;
    const lines = block.split('\n');
    const table = lines.filter(line => line.trim().startsWith('|'));
    if (table.length >= 2) {
      const cells = (line: string) => line.split('|').slice(1, -1).map(cell => cell.trim());
      const rows = table.filter((_, row) => row !== 1).map(cells);
      return <div className="markdown-table-wrap" key={index}><table className="markdown-table"><thead><tr>{(rows[0] || []).map((cell, cellIndex) => <th key={cellIndex}>{cell}</th>)}</tr></thead><tbody>{rows.slice(1).map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}</tr>)}</tbody></table></div>;
    }
    return <>{lines.filter(line => line.trim()).map((line, lineIndex) => {
      if (line.startsWith('### ')) return <h4 key={`${index}-${lineIndex}`}>{line.slice(4)}</h4>;
      if (line.startsWith('## ')) return <h3 key={`${index}-${lineIndex}`}>{line.slice(3)}</h3>;
      if (line.startsWith('# ')) return <h2 key={`${index}-${lineIndex}`}>{line.slice(2)}</h2>;
      if (/^[-*] /.test(line)) return <li key={`${index}-${lineIndex}`}>{line.slice(2)}</li>;
      return <p key={`${index}-${lineIndex}`}>{line}</p>;
    })}</>;
  })}</div>;
}

function DiagramPreview({ source }: { source: string }) {
  const edges = source.split('\n').map(line => /([^\-\s][^\-]*?)\s*--+>\s*([^\-\s][^\-]*)/.exec(line)).filter((edge): edge is RegExpExecArray => Boolean(edge)).map(edge => [edge[1].trim(), edge[2].trim()] as const);
  const nodes = Array.from(new Set(edges.flatMap(edge => edge)));
  if (!nodes.length) return null;
  const width = Math.max(360, nodes.length * 150);
  return <svg className="diagram-preview" viewBox={`0 0 ${width} 120`} role="img" aria-label="Rendered flow diagram"><defs><marker id="diagram-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#9cc6ff" /></marker></defs>{nodes.map((node, index) => { const x = 12 + index * ((width - 132) / Math.max(1, nodes.length - 1)); return <g key={node}><rect x={x} y="38" width="120" height="42" rx="9" fill="#1e3a5f" stroke="#72a9e5" /><text x={x + 60} y="64" textAnchor="middle" fill="#eaf4ff" fontSize="12">{node}</text>{index < nodes.length - 1 && <line x1={x + 120} y1="59" x2={x + ((width - 132) / Math.max(1, nodes.length - 1))} y2="59" stroke="#9cc6ff" strokeWidth="2" markerEnd="url(#diagram-arrow)" />}</g>; })}</svg>;
}

function MediaPart({ part }: { part: ContentPart }) {
  const source = part.url || part.data || '';
  if (part.type === 'image_url' || part.type === 'image') return <figure className="media-part"><img src={source} alt={part.name || 'Generated image'} /><figcaption>{part.name || 'Image'}</figcaption></figure>;
  if (part.type === 'input_audio' || part.type === 'audio') return <div className="media-part"><strong>{part.name || 'Audio'}</strong><audio controls src={source} /></div>;
  if (part.type === 'video_url' || part.type === 'video') return <figure className="media-part"><video controls src={source} /><figcaption>{part.name || 'Video'}</figcaption></figure>;
  if (part.type === 'diagram') return <details className="diagram-part" open><summary>{part.name || 'Diagram'}</summary><DiagramPreview source={part.text || ''} /><pre>{part.text}</pre></details>;
  if (part.type === 'markdown') return <RichText value={part.text || ''} />;
  if (part.type === 'artifact' || part.type === 'file') return <a className="artifact-part" href={source} download={part.name || true}>{part.name || 'Download artifact'} <small>{part.mimeType || part.type}</small></a>;
  return null;
}

function PromptCopyAction({ value, done, fail }: { value: string; done: (message: string) => void; fail: (message: string) => void }) {
  if (!value.trim()) return null;
  return <div className="response-actions prompt-copy-action"><button type="button" className="response-icon" title="Copy prompt" aria-label="Copy prompt" onClick={() => void navigator.clipboard.writeText(value).then(() => done('Prompt copied.')).catch(() => fail('Unable to copy the prompt.'))}><CopyIcon /></button></div>;
}

function MessageTimestamp({ value }: { value?: string }) {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  const pad = (number: number) => String(number).padStart(2, '0');
  return <time className="message-timestamp" dateTime={date.toISOString()}>{pad(date.getMonth() + 1)}-{pad(date.getDate())}-{date.getFullYear()}, {pad(date.getHours())}:{pad(date.getMinutes())}</time>;
}

function WorkedFor({ latencyMs }: { latencyMs?: number }) {
  if (!latencyMs || latencyMs < 1) return null;
  const seconds = Math.max(1, Math.ceil(latencyMs / 1000));
  return <small className="worked-for">Worked for {seconds}s</small>;
}

function ResponseActions({ message, done, fail }: { message: Message; done: (message: string) => void; fail: (message: string) => void }) {
  const [open, setOpen] = useState(false);
  const text = message.content.trim();
  const downloadableParts = (message.parts || []).filter(part => ['image_url', 'image', 'input_audio', 'audio', 'video_url', 'video', 'file', 'artifact'].includes(part.type));
  const copyValue = text || downloadableParts.map(part => `${part.type}: ${part.name || part.url || 'attachment'}`).join('\n');
  const download = (value: string | Blob, name: string, type?: string) => {
    const blob = value instanceof Blob ? value : new Blob([value], { type: type || 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    link.click();
    URL.revokeObjectURL(url);
  };
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(copyValue);
      done('Response copied.');
    } catch (error) {
      fail(error instanceof Error ? error.message : 'Unable to copy the response.');
    }
  };
  const downloadPart = async (part: ContentPart, index: number) => {
    const source = part.url || part.data;
    if (!source) {
      fail('This attachment has no downloadable data.');
      return;
    }
    try {
      const response = await fetch(source);
      if (!response.ok)
        throw new Error(`Attachment download failed (${response.status}).`);
      const type = part.mimeType || response.headers.get('content-type') || 'application/octet-stream';
      const extension = type.split('/')[1]?.split(';')[0] || 'bin';
      const name = part.name || `sypra-attachment-${index + 1}.${extension}`;
      download(await response.blob(), name, type);
      setOpen(false);
      done(`${name} downloaded.`);
    } catch (error) {
      fail(error instanceof Error ? error.message : 'Unable to download the attachment.');
    }
  };
  if (!copyValue && downloadableParts.length === 0)
    return null;
  return <div className="response-actions">
    <button type="button" className="response-icon" title="Copy response" aria-label="Copy response" onClick={() => void copy()}><CopyIcon /></button>
    <div className="response-download">
      <button type="button" className="response-icon" title="Download response" aria-label="Download response" aria-expanded={open} onClick={() => setOpen(value => !value)}><DownloadIcon /></button>
      {open && <div className="response-download-menu">
        {text && <><button type="button" onClick={() => { download(text, `sypra-response-${Date.now()}.txt`); setOpen(false); }}>Plain text (.txt)</button><button type="button" onClick={() => { download(`# Sypra response\n\n${text}\n`, `sypra-response-${Date.now()}.md`, 'text/markdown;charset=utf-8'); setOpen(false); }}>Markdown (.md)</button></>}
        {downloadableParts.map((part, index) => <button type="button" key={`${part.type}-${part.name || index}`} onClick={() => void downloadPart(part, index)}>Download {part.name || part.type}</button>)}
      </div>}
    </div>
  </div>;
}

function ChatPanel({ chat, draft, messages, models, activeModelIDs, collections, bind, created, append, settle, reload, onStream, done, fail }: {
  chat?: Chat;
  draft: Partial<Pick<Chat, 'projectId' | 'modelId' | 'collectionId' | 'ragMode' | 'ragTopK' | 'ragEnabled' | 'webEnabled'>>;
  messages: Message[];
  models: Model[];
  activeModelIDs: Set<string>;
  collections: Collection[];
  bind: (kind: 'modelId' | 'collectionId' | 'ragMode' | 'ragTopK' | 'ragEnabled' | 'webEnabled', value: string | number | boolean) => Promise<void>;
  created: (chat: Chat) => Promise<void>;
  append: (id: string, messages: Message[]) => void;
  settle: (id: string, response: string, sources?: string[]) => void;
  reload: (chatID: string) => Promise<void>;
  onStream: (chatID: string, event: ChatStreamEvent) => void;
  done: (message: string) => void;
  fail: (message: string) => void;
}) {
  const [prompt, setPrompt] = useState(''), [attachments, setAttachments] = useState<ContentPart[]>([]), [busy, setBusy] = useState(false), [picker, setPicker] = useState(false), [ragPicker, setRagPicker] = useState(false), [toolsEnabled, setToolsEnabled] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const binding = chat || draft;
  const selected = models.find(model => model.id === binding.modelId);
  const attachedRAG = collections.find(collection => collection.id === binding.collectionId);
  const defaultModel = models.find(model => model.default);
  const effectiveModel = selected || defaultModel;
  const modelAvailable = Boolean(effectiveModel && (effectiveModel.provider || effectiveModel.endpoint || activeModelIDs.has(effectiveModel.id)));
  const isWorking = messages.some(message => message.pending || message.status === 'working');
  const cliLocked = Boolean(chat?.cliSessionActive);
  const hasPrompt = Boolean(prompt.trim());
  const cancel = async () => { if (!chat || !isWorking)
    return; try {
    await api(`chats/${chat.id}/cancel`, { method: 'POST' });
    await reload(chat.id);
  }
  catch (e) {
    fail((e as Error).message);
  } };
  useEffect(() => { const element = scrollRef.current; if (element)
    element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' }); }, [messages.length, messages.at(-1)?.content]);
  useEffect(() => {
    if (!chat || !isWorking)
      return;
    const events = new EventSource(`${API}chats/${chat.id}/events`);
    events.onmessage = event => {
      try {
        onStream(chat.id, JSON.parse(event.data) as ChatStreamEvent);
      } catch {
        // A malformed event must not tear down a running chat; polling below
        // remains a durable recovery path.
      }
    };
    return () => events.close();
  }, [chat?.id, isWorking, onStream]);
  useEffect(() => { if (!chat || !isWorking)
    return; const timer = window.setInterval(() => { void reload(chat.id).catch(() => undefined); }, 5000); return () => window.clearInterval(timer); }, [chat, isWorking, reload]);
  useEffect(() => { const key = (event: globalThis.KeyboardEvent) => { if (event.key === 'Escape' && isWorking) {
    event.preventDefault();
    void cancel();
  } }; window.addEventListener('keydown', key); return () => window.removeEventListener('keydown', key); }, [isWorking, chat?.id]);
  const send = async (event: FormEvent) => {
    event.preventDefault();
    const text = prompt.trim();
    if (!text || busy || isWorking || cliLocked || !modelAvailable)
      return;
    setBusy(true);
    let target = chat;
    try {
      if (!target) {
        target = await api<Chat>('chats', { method: 'POST', body: JSON.stringify(draft) });
        await created(target);
      }
      const stamp = Date.now();
      append(target.id, [{ id: `pending-user-${stamp}`, role: 'user', content: text, pending: true }, { id: `pending-assistant-${stamp}`, role: 'assistant', content: 'Working…', pending: true }]);
      setPrompt('');
      setAttachments([]);
      await api<{
        status: string;
      }>('chat', { method: 'POST', body: JSON.stringify({ chatID: target.id, prompt: text, parts: attachments, temperature: .2, ragEnabled: binding.ragEnabled ?? true, webSearch: binding.webEnabled ? true : undefined, toolsEnabled, async: true }) });
      await reload(target.id);
    }
    catch (e) {
      if (target)
        settle(target.id, `Unable to get a response: ${(e as Error).message}`);
      fail((e as Error).message);
    }
    finally {
      setBusy(false);
    }
  };
  const keyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  } };
  const useRAG = (collectionID: string) => {
    void bind('collectionId', collectionID);
    setRagPicker(false);
  };
  return (
    <div className="chat-view">
      <div className="conversation" ref={scrollRef}>
        {messages.length ? (
          messages.map(message => (
            <article
              className={`message ${message.role}${message.pending || message.status === 'working' ? ' pending' : ''}`}
              key={message.id}
            >
              <div>
				{message.role === 'assistant' && !message.pending && message.status === 'completed' && <WorkedFor latencyMs={message.latencyMs} />}
                <RichText value={message.content} />
                {message.parts?.length ? <div className="content-parts">{message.parts.map((part, index) => <MediaPart key={index} part={part} />)}</div> : null}
                {message.sources?.length ? <div className="sources">Sources: {message.sources.map(source => <span key={source}>{source.split(/[\\/]/).pop()}</span>)}</div> : null}
				<div className="message-footer"><MessageTimestamp value={message.createdAt} />{message.role === 'user' && <PromptCopyAction value={message.content} done={done} fail={fail} />}{message.role === 'assistant' && !message.pending && message.status !== 'working' && <ResponseActions message={message} done={done} fail={fail} />}</div>
              </div>
            </article>
          ))
        ) : (
          <div className="welcome">
            <img className="welcome-mark logo-mark" src="/sypra.ico" alt="Sypra logo" />
            <h1>How can I help you today?</h1>
            <p>Choose a model below, attach knowledge when needed, and start a conversation.</p>
          </div>
        )}
      </div>
      <form className="composer" onSubmit={send}>
        <textarea
          value={prompt}
          onChange={event => setPrompt(event.target.value)}
          onKeyDown={keyDown}
          placeholder={cliLocked ? 'This chat is active in the CLI session.' : isWorking ? 'This chat is generating a response...' : 'Message Sypra...'}
          aria-label="Message"
          disabled={isWorking || cliLocked}
        />
        <div className="composer-tools">
          <label className="chip icon-button attachment-button" title="Attach files" aria-label="Attach files"><PaperclipIcon /><input disabled={cliLocked} type="file" multiple onChange={async event => { try { setAttachments(await attachmentParts(Array.from(event.target.files || []))); } catch (error) { fail(error instanceof Error ? error.message : String(error)); } }} /></label>
          <button type="button" className="chip" disabled={isWorking || cliLocked} title={cliLocked ? 'This chat is active in the CLI session' : isWorking ? 'Wait for the active generation to complete before changing models' : 'Choose a model'} onClick={() => setPicker(value => !value)}>
            {selected?.name || defaultModel?.name || 'No active model'} <span>⌄</span>
          </button>
          <button type="button" className={`chip rag-binding toggle-button ${attachedRAG && binding.ragEnabled !== false ? 'enabled' : ''}`} disabled={isWorking || (!attachedRAG && !collections.length)} aria-label={attachedRAG ? (binding.ragEnabled !== false ? `Disable ${attachedRAG.name}` : `Enable ${attachedRAG.name}`) : 'Attach a RAG collection'} title={attachedRAG ? (binding.ragEnabled !== false ? `${attachedRAG.name} is enabled — click to disable for the next message` : `${attachedRAG.name} is disabled — click to enable for the next message`) : 'Attach a RAG collection'} onClick={() => attachedRAG ? void bind('ragEnabled', binding.ragEnabled === false) : setRagPicker(value => !value)}><RAGIcon /><span>{attachedRAG ? attachedRAG.name : 'Add RAG'}</span></button>
          <button type="button" className={`chip icon-button toggle-button ${binding.webEnabled ? 'enabled' : ''}`} disabled={isWorking} aria-label={binding.webEnabled ? 'Disable web access' : 'Enable web access'} title={binding.webEnabled ? 'Web access enabled — click to disable' : 'Web access disabled — click to enable'} onClick={() => void bind('webEnabled', !binding.webEnabled)}>
            <GlobeIcon />
          </button>
          <button type="button" className={`chip icon-button toggle-button ${toolsEnabled ? 'enabled' : ''}`} disabled={isWorking} aria-label={toolsEnabled ? 'Disable model tools' : 'Enable model tools'} title={toolsEnabled ? 'Model tools enabled — click to disable' : 'Model tools disabled — click to enable'} onClick={() => setToolsEnabled(value => !value)}>
            <ToolsIcon />
          </button>
        </div>
			{attachments.length > 0 && <div className="attachment-previews">{attachments.map((part, index) => <div className="attachment-preview" key={`${part.name || part.type}-${index}`}>{part.type === 'image_url' ? <img src={part.data || part.url} alt={part.name || 'Attached image'} /> : <span>{part.type === 'input_audio' ? 'Audio' : part.type === 'video_url' ? 'Video' : 'Document'} · {part.name || 'attachment'}</span>}<button type="button" aria-label={`Remove ${part.name || 'attachment'}`} onClick={() => setAttachments(current => current.filter((_, attachmentIndex) => attachmentIndex !== index))}>×</button></div>)}</div>}
        {picker && <div className="model-picker">{defaultModel && (defaultModel.provider || defaultModel.endpoint || activeModelIDs.has(defaultModel.id)) && <button type="button" onClick={() => { void bind('modelId', ''); setPicker(false); }}>Use default: {defaultModel.name}</button>}{models.filter(model => Boolean(model.provider) || Boolean(model.endpoint) || activeModelIDs.has(model.id)).map(model => <button type="button" onClick={() => { void bind('modelId', model.id); setPicker(false); }} key={model.id}>{model.name}<small>{model.kind === 'base' ? 'Base model' : 'Fine-tuned model'}</small></button>)}</div>}
        {ragPicker && <div className="rag-picker"><p className="rag-picker-title">Attach a RAG collection</p>{collections.length ? collections.map(collection => <button type="button" key={collection.id} onClick={() => useRAG(collection.id)}><b>{collection.name}</b><small>Attach this collection to the current chat</small></button>) : <p>No RAG collections yet.</p>}</div>}
        {isWorking && <button type="button" className="cancel-generation" onClick={() => void cancel()}>Stop (Esc)</button>}
        <button className="send" aria-label="Send message" title="Send message" disabled={!hasPrompt || busy || isWorking || cliLocked || !modelAvailable}><ArrowUpIcon /></button>
        <small className={`composer-hint${isWorking || cliLocked || !modelAvailable ? ' generation-lock' : ''}`}>{cliLocked ? `CLI session is active in ${chat?.workspacePath}. Run sypra chat --session ${chat?.id} --close from that folder to unlock this chat.` : isWorking ? 'Generation in progress - this chat is locked until the response completes.' : attachedRAG ? `${attachedRAG.name} is attached; retrieval is ${binding.ragEnabled !== false ? 'on' : 'off'} for the next message.` : !modelAvailable ? 'Start a local model from the Models catalog before sending a chat.' : 'Enter to send - Shift+Enter for a new line'}</small>
      </form>
    </div>
  );
}

function RagPanel({ collections, chats, openEditor, openChat, createRAGChat, onChanged, done, fail }: {
  collections: Collection[];
  chats: Chat[];
  openEditor: (collection?: Collection) => void;
  openChat: (chat: Chat) => Promise<void>;
  createRAGChat: (collectionID: string) => Promise<void>;
  onChanged: () => Promise<void>;
  done: (message: string) => void;
  fail: (message: string) => void;
}) {
  const [selected, setSelected] = useState<Collection>();
  const [documents, setDocuments] = useState<RAGDocument[]>([]);
  const [query, setQuery] = useState('');
  const [metadata, setMetadata] = useState('');
  const [rerank, setRerank] = useState(true);
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [rename, setRename] = useState<RAGDocument>();
  const [renameValue, setRenameValue] = useState('');
  const [removeCandidate, setRemoveCandidate] = useState<RAGDocument>();
	const [replaceCandidate, setReplaceCandidate] = useState<RAGDocument>();
	const [replaceFile, setReplaceFile] = useState<File>();
  const [evaluationName, setEvaluationName] = useState('');
  const [evaluationQuery, setEvaluationQuery] = useState('');
  const [evaluationDocumentID, setEvaluationDocumentID] = useState('');
  const [evaluations, setEvaluations] = useState<EvaluationSet[]>([]);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const loadLibrary = async (collection: Collection) => {
    setSelected(collection);
    setResults([]);
    try {
      const [nextDocuments, nextEvaluations] = await Promise.all([
        api<RAGDocument[]>(`rag/collections/${collection.id}/documents`),
        api<EvaluationSet[]>(`rag/evaluations?collectionId=${encodeURIComponent(collection.id)}`),
      ]);
      setDocuments(list(nextDocuments));
      setEvaluations(list(nextEvaluations));
    } catch (error) { fail((error as Error).message); }
  };
  const refreshLibrary = async () => { if (selected) await loadLibrary(selected); await onChanged(); };
  const removeDocument = async () => {
    if (!removeCandidate) return;
    try { await api(`rag/documents/${removeCandidate.id}`, { method: 'DELETE' }); setRemoveCandidate(undefined); await refreshLibrary(); done('Document removed from the RAG index.'); } catch (error) { fail((error as Error).message); }
  };
  const reindex = async (document: RAGDocument) => {
    try { await api(`rag/documents/${document.id}/reindex`, { method: 'POST', body: JSON.stringify({}) }); await refreshLibrary(); done(`${document.name} re-indexed.`); } catch (error) { fail((error as Error).message); }
  };
  const replaceDocument = async (event: FormEvent) => {
    event.preventDefault();
    if (!replaceCandidate || !replaceFile) return;
    try {
      const upload = new FormData(); upload.append('file', replaceFile);
      const source = await api<{ source: string; path: string }>('rag/documents/upload', { method: 'POST', body: upload });
      await api(`rag/documents/${replaceCandidate.id}/reindex`, { method: 'POST', body: JSON.stringify({ document: source }) });
      setReplaceCandidate(undefined); setReplaceFile(undefined); await refreshLibrary(); done(`${replaceCandidate.name} replaced and re-indexed.`);
    } catch (error) { fail((error as Error).message); }
  };
  const saveRename = async () => {
    if (!rename || !renameValue.trim()) return;
    try { await api(`rag/documents/${rename.id}`, { method: 'PATCH', body: JSON.stringify({ name: renameValue.trim() }) }); setRename(undefined); await refreshLibrary(); done('Document renamed.'); } catch (error) { fail((error as Error).message); }
  };
  const inspect = async (event: FormEvent) => {
    event.preventDefault();
    if (!selected || !query.trim()) return;
    try {
      const filters = metadata.trim() ? Object.fromEntries(metadata.split(',').map(value => value.split('=').map(item => item.trim())).filter(pair => pair.length === 2)) : undefined;
      const response = await api<{ results: RetrievalResult[] }>('rag/query', { method: 'POST', body: JSON.stringify({ collectionID: selected.id, query, limit: 6, metadata: filters, rerank, includeNeighbors: true }) });
      setResults(list(response.results));
    } catch (error) { fail((error as Error).message); }
  };
  const createEvaluation = async (event: FormEvent) => {
    event.preventDefault();
    if (!selected || !evaluationName.trim() || !evaluationQuery.trim() || !evaluationDocumentID) return;
    try {
      const set = await api<EvaluationSet>('rag/evaluations', { method: 'POST', body: JSON.stringify({ collectionID: selected.id, name: evaluationName, cases: [{ query: evaluationQuery, relevantDocumentIds: [evaluationDocumentID] }] }) });
      setEvaluations(current => [...current, set]); setEvaluationName(''); setEvaluationQuery(''); done('RAG evaluation set created.');
    } catch (error) { fail((error as Error).message); }
  };
  const runEvaluation = async (evaluation: EvaluationSet) => {
    try {
      const run = await api<EvaluationRun>(`rag/evaluations/${evaluation.id}/run`, { method: 'POST', body: JSON.stringify({ limit: 5, rerank }) });
      setRuns(current => [run, ...current]); done(`Evaluation complete: ${(run.metrics.hitRate * 100).toFixed(0)}% hit rate.`);
    } catch (error) { fail((error as Error).message); }
  };
  return <div className="grid rag-catalog">
    <Card title="RAG collections">
      <p className="hint">Collections are local knowledge sources. Add or edit their documents in an overlay; choose one before a chat begins to keep retrieval reproducible.</p>
      {collections.length ? <div className="rag-catalog-list">{collections.map(collection => {
        const relatedChats = chats.filter(chat => chat.collectionId === collection.id);
        return <article className="rag-catalog-item" key={collection.id}>
          <div><h3>{collection.name}</h3><p>{collection.documentCount} documents · {collection.embeddingModel || 'Local embeddings'}</p></div>
          <div className="rag-catalog-meta"><span>{relatedChats.length} related chats</span><span>{collection.chunkSize} {collection.chunkUnit || 'character'} target chunks</span></div>
          {!collection.chunkUnit && <p className="hint">Legacy character-based index. Re-upload documents to rebuild it with transformer embeddings and token chunks.</p>}
          <div className="rag-catalog-actions"><button type="button" className="secondary" onClick={() => openEditor(collection)}>Add documents</button><button type="button" className="secondary" onClick={() => void loadLibrary(collection)}>Document library</button><button type="button" onClick={() => void createRAGChat(collection.id)}>New RAG chat</button></div>
        </article>;
      })}</div> : <p className="empty">No RAG collections yet. Use the + button beside RAGs to create one.</p>}
    </Card>
    <Card title="Chats using RAG">
      {chats.filter(chat => chat.collectionId).length ? <div className="rag-related-chats">{chats.filter(chat => chat.collectionId).map(chat => {
        const collection = collections.find(item => item.id === chat.collectionId);
        return <button type="button" key={chat.id} onClick={() => void openChat(chat)}><span>{chat.title}</span><small>{collection?.name || 'Deleted RAG collection'}</small></button>;
      })}</div> : <p className="empty">No chats are bound to a RAG collection yet.</p>}
    </Card>
    {selected && <Card title={`Document library · ${selected.name}`}>
      <p className="hint">Every source is independently inspectable and re-indexable. Existing chunks stay available if a URL refresh or replacement fails.</p>
      <div className="rag-document-list">{documents.map(document => <article key={document.id} className="rag-document-row"><div><strong>{document.name}</strong><small>{document.contentType || 'document'} · {document.chunkCount} chunks · extraction: {document.extractionStatus}{document.nextRefreshAt ? ` · refreshes ${new Date(document.nextRefreshAt).toLocaleString()}` : ''}</small>{document.extractionError && <small className="error-text">{document.extractionError}</small>}</div><div className="rag-document-actions"><button type="button" onClick={() => { setRename(document); setRenameValue(document.name); }}>Rename</button><button type="button" onClick={() => { setReplaceCandidate(document); setReplaceFile(undefined); }}>Replace</button><button type="button" onClick={() => void reindex(document)}>Re-index</button>{document.url && <button type="button" onClick={() => void api(`rag/documents/${document.id}/refresh`, { method: 'POST', body: JSON.stringify({}) }).then(() => refreshLibrary()).catch(error => fail((error as Error).message))}>Refresh URL</button>}<button type="button" className="danger" onClick={() => setRemoveCandidate(document)}>Remove</button></div></article>)}</div>
      {!documents.length && <p className="empty">No documents found. Add one from the RAG editor.</p>}
      <form className="rag-inspector" onSubmit={inspect}><h3>Source preview</h3><label>Test query<input value={query} onChange={event => setQuery(event.target.value)} placeholder="Ask a retrieval question" /></label><div className="rag-controls"><label>Metadata filters <input value={metadata} onChange={event => setMetadata(event.target.value)} placeholder="department=legal, year=2026" /></label><label className="check"><input type="checkbox" checked={rerank} onChange={event => setRerank(event.target.checked)} /> Local reranking</label><button>Inspect retrieval</button></div></form>
      <div className="rag-results">{results.map(result => <article key={result.chunk.id} className="rag-result"><header><strong>{result.document?.name || result.chunk.source}</strong><span>score {result.score.toFixed(3)} · semantic {result.semanticScore.toFixed(3)} · lexical {result.lexicalScore.toFixed(3)} · {result.location}</span></header><p>{result.chunk.content}</p>{result.previous && <details><summary>Previous chunk</summary><p>{result.previous.content}</p></details>}{result.next && <details><summary>Next chunk</summary><p>{result.next.content}</p></details>}</article>)}</div>
      <form className="rag-evaluation" onSubmit={createEvaluation}><h3>Retrieval evaluation</h3><div className="rag-controls"><label>Set name<input required value={evaluationName} onChange={event => setEvaluationName(event.target.value)} placeholder="Support retrieval checks" /></label><label>Query<input required value={evaluationQuery} onChange={event => setEvaluationQuery(event.target.value)} placeholder="Known-answer query" /></label><label>Relevant document<select required value={evaluationDocumentID} onChange={event => setEvaluationDocumentID(event.target.value)}><option value="">Choose source</option>{documents.map(document => <option value={document.id} key={document.id}>{document.name}</option>)}</select></label><button>Create set</button></div></form>
      {evaluations.map(evaluation => <div className="rag-evaluation-row" key={evaluation.id}><span>{evaluation.name} · {evaluation.cases.length} case{evaluation.cases.length === 1 ? '' : 's'}</span><button type="button" onClick={() => void runEvaluation(evaluation)}>Run @5</button></div>)}
      {runs.map(run => <p className="hint" key={run.id}>Latest evaluation: hit rate {(run.metrics.hitRate * 100).toFixed(1)}%, recall@K {(run.metrics.recallAtK * 100).toFixed(1)}%, precision@K {(run.metrics.precisionAtK * 100).toFixed(1)}%, MRR {run.metrics.mrr.toFixed(3)}.</p>)}
    </Card>}
    {rename && <div className="modal-layer" role="presentation" onMouseDown={() => setRename(undefined)}><section className="modal small-modal" role="dialog" aria-modal="true" aria-label="Rename document" onMouseDown={event => event.stopPropagation()}><div className="modal-heading"><h2>Rename document</h2><button type="button" onClick={() => setRename(undefined)}>×</button></div><label>Name<input autoFocus value={renameValue} onChange={event => setRenameValue(event.target.value)} /></label><div className="modal-actions"><button type="button" className="secondary" onClick={() => setRename(undefined)}>Cancel</button><button type="button" onClick={() => void saveRename()}>Save</button></div></section></div>}
    {removeCandidate && <div className="modal-layer" role="presentation" onMouseDown={() => setRemoveCandidate(undefined)}><section className="modal small-modal" role="dialog" aria-modal="true" aria-label="Remove document" onMouseDown={event => event.stopPropagation()}><div className="modal-heading"><h2>Remove document</h2><button type="button" onClick={() => setRemoveCandidate(undefined)}>×</button></div><p className="modal-copy">Remove {removeCandidate.name} and its {removeCandidate.chunkCount} indexed chunks? This cannot be undone from the library.</p><div className="modal-actions"><button type="button" className="secondary" onClick={() => setRemoveCandidate(undefined)}>Cancel</button><button type="button" className="danger" onClick={() => void removeDocument()}>Remove</button></div></section></div>}
    {replaceCandidate && <div className="modal-layer" role="presentation" onMouseDown={() => setReplaceCandidate(undefined)}><section className="modal small-modal" role="dialog" aria-modal="true" aria-label="Replace document" onMouseDown={event => event.stopPropagation()}><div className="modal-heading"><h2>Replace document</h2><button type="button" onClick={() => setReplaceCandidate(undefined)}>×</button></div><p className="modal-copy">The new file replaces {replaceCandidate.name}; prior chunks and vectors are removed only after the replacement has been indexed.</p><form onSubmit={replaceDocument}><label>Replacement file<input required type="file" onChange={event => setReplaceFile(event.target.files?.[0])} /></label><div className="modal-actions"><button type="button" className="secondary" onClick={() => setReplaceCandidate(undefined)}>Cancel</button><button disabled={!replaceFile}>Replace and index</button></div></form></section></div>}
  </div>;
}

function RagEditorOverlay({ editing, onClose, onCreated, done, fail }: {
  editing?: Collection;
  onClose: () => void;
  onCreated: () => void;
  done: (message: string) => void;
  fail: (message: string) => void;
}) {
  const [name, setName] = useState('');
	const [routingTags, setRoutingTags] = useState('');
  const [text, setText] = useState('');
  const [source, setSource] = useState('notes.txt');
  const [folder, setFolder] = useState('');
	const [url, setURL] = useState('');
	const [refreshMinutes, setRefreshMinutes] = useState(0);
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState<'paste' | 'files' | 'folder' | 'url'>('paste');
  const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    try {
      const documents = mode === 'paste'
        ? [{ source, text }]
        : mode === 'folder'
          ? [{ folder }]
			: mode === 'url'
				? [{ url, refreshIntervalMinutes: refreshMinutes }]
          : await Promise.all(files.map(async file => {
            const upload = new FormData();
            upload.append('file', file);
            return api<{ source: string; path: string }>('rag/documents/upload', { method: 'POST', body: upload });
          }));
      if (editing) {
        await api(`rag/collections/${editing.id}/documents`, { method: 'POST', body: JSON.stringify({ documents }) });
        done(`Added documents to ${editing.name}.`);
      } else {
        await api('rag/collections', { method: 'POST', body: JSON.stringify({ name, embeddingModel: 'all-minilm-l6-v2-token-mean', chunkSize: 192, chunkOverlap: 32, routingTags: routingTags.split(',').map(tag => tag.trim()).filter(Boolean), documents }) });
        done('RAG collection indexed with local MiniLM embeddings.');
        onCreated();
      }
      onClose();
    } catch (e) {
      fail((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return <div className="modal-layer" role="presentation" onMouseDown={onClose}>
    <section className="modal rag-editor-modal" role="dialog" aria-modal="true" aria-labelledby="rag-editor-title" onMouseDown={event => event.stopPropagation()}>
      <div className="modal-heading"><div><p>RAG collection</p><h2 id="rag-editor-title">{editing ? `Edit ${editing.name}` : 'Create RAG collection'}</h2></div><button type="button" aria-label="Close dialog" onClick={onClose}>×</button></div>
      <p className="modal-copy">{editing ? 'Add documents without replacing existing knowledge or embeddings.' : 'Uses local MiniLM embeddings, vector retrieval, and lexical tie-breaking.'}</p>
      <form onSubmit={submit}>
        {!editing && <><label>Name<input autoFocus required value={name} onChange={event => setName(event.target.value)} placeholder="e.g. Product documentation" /></label><label>Routing tags (optional)<input value={routingTags} onChange={event => setRoutingTags(event.target.value)} placeholder="support, product, legal" /></label></>}
        <label>Source type<select value={mode} onChange={event => setMode(event.target.value as typeof mode)}><option value="paste">Paste text</option><option value="files">Upload files</option><option value="folder">Trusted local folder</option><option value="url">Web URL</option></select></label>
        {mode === 'paste' && <><label>Source label<input value={source} onChange={event => setSource(event.target.value)} /></label><label>Text<textarea required value={text} onChange={event => setText(event.target.value)} /></label></>}
        {mode === 'files' && <label>Files<input type="file" multiple required accept=".docx,.txt,.md,.markdown,.csv,.tsv,.json,.jsonl,.html,.htm,.xml,.yaml,.yml,application/vnd.openxmlformats-officedocument.wordprocessingml.document" onChange={event => setFiles(Array.from(event.target.files || []))} /></label>}
        {mode === 'folder' && <label>Folder path<input required value={folder} onChange={event => setFolder(event.target.value)} placeholder="D:\knowledge-base" /></label>}
        {mode === 'url' && <div className="rag-url-fields"><label>Web URL<input required type="url" value={url} onChange={event => setURL(event.target.value)} placeholder="https://docs.example.com/page" /></label><label>Refresh every (minutes, optional)<input type="number" min="0" value={refreshMinutes} onChange={event => setRefreshMinutes(Number(event.target.value))} /></label></div>}
        <div className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button disabled={busy}>{busy ? 'Indexing…' : editing ? 'Add to RAG' : 'Set up RAG'}</button></div>
      </form>
    </section>
  </div>;
}

function ToolsPanel({ tools, connections, permissions, runs, models, projects, onChanged, done, fail }: { tools: ToolDefinition[]; connections: ToolConnection[]; permissions: ToolPermission[]; runs: AgentRun[]; models: Model[]; projects: Project[]; onChanged: () => Promise<void>; done: (message: string) => void; fail: (message: string) => void }) {
  const [connectionName, setConnectionName] = useState('');
  const [connectionKind, setConnectionKind] = useState<'mcp' | 'pipeline'>('mcp');
  const [command, setCommand] = useState('');
  const [args, setArgs] = useState('');
  const [pipelineTools, setPipelineTools] = useState<string[]>([]);
  const [policyTool, setPolicyTool] = useState('');
  const [policyModel, setPolicyModel] = useState('');
  const [policyProject, setPolicyProject] = useState('');
  const [requiresApproval, setRequiresApproval] = useState(true);
  const [runTool, setRunTool] = useState('');
  const [runModel, setRunModel] = useState('');
  const [runProject, setRunProject] = useState('');
  const [runArguments, setRunArguments] = useState('{\n  \n}');
  const [approved, setApproved] = useState(false);
  const submitConnection = async (event: FormEvent) => {
    event.preventDefault();
    try {
      await api('tools/connections', { method: 'POST', body: JSON.stringify({ name: connectionName, kind: connectionKind, command, args: args.split(/\s+/).filter(Boolean), toolIds: pipelineTools, enabled: true }) });
      setConnectionName(''); setCommand(''); setArgs(''); setPipelineTools([]); await onChanged(); done(`${connectionKind === 'mcp' ? 'MCP connection' : 'Pipeline'} added.`);
    } catch (error) { fail((error as Error).message); }
  };
  const submitPolicy = async (event: FormEvent) => {
    event.preventDefault();
    if (!policyTool) { fail('Choose a tool first.'); return; }
    try { await api('tools/permissions', { method: 'POST', body: JSON.stringify({ toolID: policyTool, modelID: policyModel, projectID: policyProject, enabled: true, requireApproval: requiresApproval }) }); await onChanged(); done('Tool permission added.'); } catch (error) { fail((error as Error).message); }
  };
  const execute = async (event: FormEvent) => {
    event.preventDefault();
    let argumentsValue: Record<string, unknown>;
    try { argumentsValue = JSON.parse(runArguments) as Record<string, unknown>; } catch { fail('Tool arguments must be valid JSON.'); return; }
    try { const run = await api<AgentRun>('agents/runs', { method: 'POST', body: JSON.stringify({ toolID: runTool, modelID: runModel, projectID: runProject, arguments: argumentsValue, approved }) }); await onChanged(); done(run.status === 'completed' ? 'Agent tool completed.' : 'Agent tool finished.'); } catch (error) { fail((error as Error).message); }
  };
  const approveRun = async (run: AgentRun) => {
    try {
      await api<AgentRun>(`agents/runs/${run.id}/approve`, { method: 'POST', body: '{}' });
      await onChanged();
      done('Tool execution approved.');
    } catch (error) { fail((error as Error).message); }
  };
  return <div className="grid tools-workspace">
    <Card title="Pending approvals"><p className="hint">Approval preserves the exact recorded model, project, tool, and arguments. Review before approving.</p><div className="agent-run-list">{runs.filter(run => run.status === 'awaiting_approval').map(run => <article key={run.id}><strong>{run.toolId}</strong><pre>{JSON.stringify(run.arguments, null, 2)}</pre><button type="button" onClick={() => void approveRun(run)}>Approve recorded request</button></article>)}{!runs.some(run => run.status === 'awaiting_approval') && <p className="empty">No tool calls are awaiting approval.</p>}</div></Card>
    <Card title="Tool catalog"><p className="hint">Tools are denied by default. Add an explicit model/project policy before an agent can call one.</p><div className="tool-catalog">{tools.map(tool => <article key={tool.id}><strong>{tool.name}</strong><small>{tool.id} · {tool.source}</small><p>{tool.description}</p></article>)}</div></Card>
    <Card title="MCP connections & pipelines"><form onSubmit={submitConnection}><label>Name<input required value={connectionName} onChange={event => setConnectionName(event.target.value)} placeholder="e.g. Local filesystem MCP" /></label><label>Extension type<select value={connectionKind} onChange={event => setConnectionKind(event.target.value as 'mcp' | 'pipeline')}><option value="mcp">MCP stdio server</option><option value="pipeline">Pipeline</option></select></label>{connectionKind === 'mcp' ? <><label>Command<input required value={command} onChange={event => setCommand(event.target.value)} placeholder="npx or mcp-server-filesystem" /></label><label>Arguments<input value={args} onChange={event => setArgs(event.target.value)} placeholder="-y package-name C:\approved-folder" /></label></> : <fieldset className="tool-checkboxes"><legend>Pipeline tools</legend>{tools.map(tool => <label key={tool.id}><input type="checkbox" checked={pipelineTools.includes(tool.id)} onChange={() => setPipelineTools(current => current.includes(tool.id) ? current.filter(id => id !== tool.id) : [...current, tool.id])} /> {tool.name}</label>)}</fieldset>}<button>Add extension</button></form><div className="tool-connection-list">{connections.map(connection => <div key={connection.id}><strong>{connection.name}</strong><small>{connection.kind} · {connection.enabled ? 'enabled' : 'disabled'}{connection.command ? ` · ${connection.command}` : ''}</small><button type="button" className="danger" onClick={() => void api(`tools/connections/${connection.id}`, { method: 'DELETE' }).then(onChanged).then(() => done('Connection removed.')).catch(error => fail((error as Error).message))}>Remove</button></div>)}</div></Card>
    <Card title="Tool permissions"><form onSubmit={submitPolicy}><label>Tool<select value={policyTool} onChange={event => setPolicyTool(event.target.value)}><option value="">Choose tool</option>{tools.map(tool => <option value={tool.id} key={tool.id}>{tool.name}</option>)}</select></label><div className="tool-policy-grid"><label>Model<select value={policyModel} onChange={event => setPolicyModel(event.target.value)}><option value="">Any model</option>{models.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}</select></label><label>Project<select value={policyProject} onChange={event => setPolicyProject(event.target.value)}><option value="">Any project</option>{projects.map(project => <option key={project.id} value={project.id}>{project.name}</option>)}</select></label></div><label className="check"><input type="checkbox" checked={requiresApproval} onChange={event => setRequiresApproval(event.target.checked)} /> Require explicit run approval</label><button>Add permission</button></form><div className="tool-permission-list">{permissions.map(permission => <p key={permission.id}><strong>{permission.toolId}</strong> · {models.find(model => model.id === permission.modelId)?.name || 'any model'} · {projects.find(project => project.id === permission.projectId)?.name || 'any project'} · {permission.requireApproval ? 'approval required' : 'auto-approved'}</p>)}</div></Card>
    <Card title="Agent task runner"><p className="hint">This invokes a configured function with an explicit model/project scope. Use it to validate permissions and extension results before enabling an autonomous model workflow.</p><form onSubmit={execute}><label>Tool<select required value={runTool} onChange={event => setRunTool(event.target.value)}><option value="">Choose tool</option>{tools.map(tool => <option value={tool.id} key={tool.id}>{tool.name}</option>)}</select></label><div className="tool-policy-grid"><label>Model<select value={runModel} onChange={event => setRunModel(event.target.value)}><option value="">No model scope</option>{models.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}</select></label><label>Project<select value={runProject} onChange={event => setRunProject(event.target.value)}><option value="">No project scope</option>{projects.map(project => <option key={project.id} value={project.id}>{project.name}</option>)}</select></label></div><label>Function arguments (JSON)<textarea value={runArguments} onChange={event => setRunArguments(event.target.value)} /></label><label className="check"><input type="checkbox" checked={approved} onChange={event => setApproved(event.target.checked)} /> I approve this tool execution</label><button>Run agent tool</button></form><div className="agent-run-list">{runs.map(run => <details key={run.id}><summary>{run.toolId} · {run.status}</summary>{run.output && <pre>{run.output}</pre>}{run.error && <p className="error-text">{run.error}</p>}</details>)}</div></Card>
  </div>;
}

function MediaPanel({ jobs, onChanged, done, fail }: { jobs: MediaJob[]; onChanged: () => Promise<void>; done: (message: string) => void; fail: (message: string) => void }) {
  const [kind, setKind] = useState<MediaJob['kind']>('image_generation');
  const [prompt, setPrompt] = useState('');
  const [file, setFile] = useState<File>();
  const [preview, setPreview] = useState('');
  const [busy, setBusy] = useState(false);
  const requiresInput = kind !== 'image_generation';
  const chooseFile = (next?: File) => {
    setFile(next);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(next?.type.startsWith('image/') ? URL.createObjectURL(next) : '');
  };
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (requiresInput && !file) { fail('Choose an input file for this workflow.'); return; }
    setBusy(true);
    try {
      let inputPath = '', inputName = '';
      if (file) { const upload = new FormData(); upload.append('file', file); const saved = await api<{ path: string; name: string }>('media/upload', { method: 'POST', body: upload }); inputPath = saved.path; inputName = saved.name; }
      await api<MediaJob>('media/jobs', { method: 'POST', body: JSON.stringify({ kind, prompt, inputPath, inputName }) });
      setPrompt(''); chooseFile(undefined); await onChanged(); done('Media workflow queued.');
    } catch (error) { fail((error as Error).message); } finally { setBusy(false); }
  };
  const cancel = async (job: MediaJob) => { try { await api(`media/jobs/${job.id}/cancel`, { method: 'POST' }); await onChanged(); done('Media workflow cancelled.'); } catch (error) { fail((error as Error).message); } };
  return <div className="grid media-workspace">
    <Card title="Create or process media"><p className="hint">All workflows are local commands. Configure image generation/editing or transcription commands in the API environment; OCR uses local Tesseract when installed.</p><form onSubmit={submit}><label>Workflow<select value={kind} onChange={event => setKind(event.target.value as MediaJob['kind'])}><option value="image_generation">Generate image</option><option value="image_edit">Edit image</option><option value="ocr">OCR image or document</option><option value="transcription">Transcribe audio or video</option></select></label>{(kind === 'image_generation' || kind === 'image_edit') && <label>Prompt<textarea required value={prompt} onChange={event => setPrompt(event.target.value)} placeholder={kind === 'image_edit' ? 'Describe the edit to apply' : 'Describe the image to generate'} /></label>}<label>{requiresInput ? 'Input file' : 'Optional reference image'}<input type="file" required={requiresInput} accept={kind === 'transcription' ? 'audio/*,video/*' : kind === 'ocr' ? 'image/*,.pdf' : 'image/*'} onChange={event => chooseFile(event.target.files?.[0])} /></label>{preview && <img className="media-input-preview" src={preview} alt="Selected input preview" />}<button disabled={busy}>{busy ? 'Queueing…' : 'Start workflow'}</button></form></Card>
    <Card title="Media jobs"><div className="media-jobs">{jobs.length ? jobs.map(job => <article className={`media-job ${job.status}`} key={job.id}><header><strong>{job.kind.replace('_', ' ')}</strong><span>{job.status} · {Math.round(job.progress * 100)}%</span></header>{job.prompt && <p>{job.prompt}</p>}{job.inputName && <small>Input: {job.inputName}</small>}{job.error && <small className="error-text">{job.error}</small>}{job.text && <details><summary>Extracted text</summary><pre>{job.text}</pre></details>}{job.outputUrl && job.outputMime?.startsWith('image/') && <img src={`${API_ORIGIN}${job.outputUrl}`} alt={`${job.kind} output`} />}{job.outputUrl && !job.outputMime?.startsWith('image/') && <a href={`${API_ORIGIN}${job.outputUrl}`} download>Download output</a>}{(job.status === 'queued' || job.status === 'running') && <button type="button" onClick={() => void cancel(job)}>Cancel</button>}</article>) : <p className="empty">No media jobs yet.</p>}</div></Card>
  </div>;
}

function TrainingPanel({ jobs, baseModels, setupOnly = false, onQueued, done, fail }: {
  jobs: Job[];
  baseModels: Model[];
  setupOnly?: boolean;
  onQueued?: () => void;
  done: (message: string) => void;
  fail: (message: string) => void;
}) {
  const [form, setForm] = useState({ name: '', baseModel: '', method: 'lora', computeBackend: 'auto', gpuDevices: '', workerEndpoints: '', epochs: 3, batchSize: 1, learningRate: .0002, loraRank: 16, loraAlpha: 32, checkpointEvery: 1, earlyStoppingPatience: 0, targets: 'all', weightDecay: .01, maxGradientNorm: 1, maxSequenceLength: 0 }), [dataset, setDataset] = useState<File>();
  const submit = async (event: FormEvent) => { event.preventDefault(); if (!dataset)
    return fail('Upload a CSV dataset.'); try {
    const upload = new FormData();
    upload.append('file', dataset);
    const stored = await api<{
      path: string;
    }>('training/datasets/upload', { method: 'POST', body: upload });
    const base = baseModels.find(model => model.id === form.baseModel);
    const targetModules = form.targets.trim() === '' || form.targets.trim().toLowerCase() === 'all' ? [] : form.targets.split(',').map(target => target.trim()).filter(Boolean);
    const gpuDevices = form.gpuDevices.split(',').map(value => Number(value.trim())).filter(value => Number.isInteger(value) && value >= 0);
    const workerEndpoints = form.workerEndpoints.split(',').map(value => value.trim()).filter(Boolean);
    await api('training/jobs', { method: 'POST', body: JSON.stringify({ name: form.name, baseModel: base?.path || form.baseModel, datasetPath: stored.path, method: form.method, parameters: { computeBackend: form.computeBackend, gpuDevices, workerEndpoints, epochs: form.epochs, batchSize: form.batchSize, learningRate: form.learningRate, loraRank: form.loraRank, loraAlpha: form.loraAlpha, checkpointEvery: form.checkpointEvery, earlyStoppingPatience: form.earlyStoppingPatience, targetModules, weightDecay: form.weightDecay, maxGradientNorm: form.maxGradientNorm, maxSequenceLength: form.maxSequenceLength } }) });
    done('Training job queued.');
    onQueued?.();
  }
  catch (e) {
    fail((e as Error).message);
  } };
  return <div className={`grid${setupOnly ? ' training-setup-grid' : ''}`}><Card title="Fine-tune an SLM"><p className="hint">Upload CSV with instruction/response columns, or JSONL using instruction/response, prompt/completion, or chat messages ending in an assistant response.</p><form onSubmit={submit}><label>Run name<input required value={form.name} onChange={e => setForm({ ...form, name: e.target.value })}/></label><label>Base model<select required value={form.baseModel} onChange={e => setForm({ ...form, baseModel: e.target.value })}><option value="">Select a base model</option>{baseModels.map(model => <option value={model.id} key={model.id}>{model.name}</option>)}</select></label><label>Training dataset<input required type="file" accept=".csv,.jsonl,.json,text/csv,application/json" onChange={e => setDataset(e.target.files?.[0])}/></label><label>Method<select value={form.method} onChange={e => setForm({ ...form, method: e.target.value })}><option value="lora">LoRA</option><option value="qlora">QLoRA</option></select></label><label>Compute backend<select value={form.computeBackend} onChange={e => setForm({ ...form, computeBackend: e.target.value })}><option value="auto">Auto (GPU if available)</option><option value="cpu">CPU</option><option value="cuda">NVIDIA CUDA</option></select></label><label>Local GPU device IDs<input value={form.gpuDevices} onChange={e => setForm({ ...form, gpuDevices: e.target.value })} placeholder="0,1"/></label><label>Remote worker URLs<input value={form.workerEndpoints} onChange={e => setForm({ ...form, workerEndpoints: e.target.value })} placeholder="http://node-a:8091,http://node-b:8091"/></label><label>Epochs<input type="number" min="1" value={form.epochs} onChange={e => setForm({ ...form, epochs: +e.target.value })}/></label><label>Batch size<input type="number" min="1" value={form.batchSize} onChange={e => setForm({ ...form, batchSize: +e.target.value })}/></label><label>Learning rate<input type="number" step="0.00001" value={form.learningRate} onChange={e => setForm({ ...form, learningRate: +e.target.value })}/></label><label>LoRA targets<input value={form.targets} onChange={e => setForm({ ...form, targets: e.target.value })} placeholder="all or q_proj,v_proj,down_proj"/></label><label>LoRA rank<input type="number" min="1" value={form.loraRank} onChange={e => setForm({ ...form, loraRank: +e.target.value })}/></label><label>LoRA alpha<input type="number" min="1" value={form.loraAlpha} onChange={e => setForm({ ...form, loraAlpha: +e.target.value })}/></label><label>Weight decay<input type="number" min="0" step="0.001" value={form.weightDecay} onChange={e => setForm({ ...form, weightDecay: +e.target.value })}/></label><label>Gradient clipping norm<input type="number" min="0" step="0.1" value={form.maxGradientNorm} onChange={e => setForm({ ...form, maxGradientNorm: +e.target.value })}/></label><label>Max sequence tokens (0 = model context)<input type="number" min="0" value={form.maxSequenceLength} onChange={e => setForm({ ...form, maxSequenceLength: +e.target.value })}/></label><label>Checkpoint every epoch(s)<input type="number" min="1" value={form.checkpointEvery} onChange={e => setForm({ ...form, checkpointEvery: +e.target.value })}/></label><label>Early stopping patience<input type="number" min="0" value={form.earlyStoppingPatience} onChange={e => setForm({ ...form, earlyStoppingPatience: +e.target.value })}/></label><button>Start fine-tuning</button></form></Card><Card title="Runs & validation"><Table rows={jobs} cols={['name', 'method', 'status', 'phase', 'progress', 'epoch', 'step', 'tokens', 'tokensPerSecond', 'trainLoss', 'evalLoss', 'error']}/></Card></div>;
}
function ModelsPanel({ models, runners, setDefaultBaseModel, startModel, stopModel, done, fail }: {
  models: Model[];
  runners: RunnerStatus[];
  setDefaultBaseModel: (modelID: string) => Promise<void>;
  startModel: (modelID: string) => Promise<void>;
  stopModel: (modelID: string) => Promise<void>;
  done: (message: string) => void;
  fail: (message: string) => void;
}) {
  const fineTunedModels = models.filter(model => model.kind === 'slm');
  const baseModels = models.filter(model => model.kind === 'base');
  const [local, setLocal] = useState({ name: '', path: '', family: 'qwen' });
  const [catalogDirectory, setCatalogDirectory] = useState('');
  const [catalogEntries, setCatalogEntries] = useState<LocalCatalogEntry[]>([]);
  const [catalogBusy, setCatalogBusy] = useState(false);
  const runnerFor = (modelID: string) => runners.find(runner => runner.modelId === modelID);
  const runnerControl = (model: Model) => {
    if (model.provider)
      return <span className="model-status external">Cloud</span>;
	if (model.endpoint)
	  return <span className="model-status external">Endpoint</span>;
    const runner = runnerFor(model.id);
    if (runner?.state === 'running' || runner?.state === 'starting')
      return <><span className={`model-status ${runner.state}`}>{runner.state}{runner.runtime ? ` · ${runner.runtime}` : ''}</span><button type="button" className="make-default" onClick={() => void stopModel(model.id)}>Stop</button></>;
    return <button type="button" className="make-default" onClick={() => void startModel(model.id)}>Start</button>;
  };
  const [name, setName] = useState(''), [kind, setKind] = useState<'base' | 'slm'>('base'), [endpoint, setEndpoint] = useState(''), [family, setFamily] = useState('generic'), [modalities, setModalities] = useState('text'), [file, setFile] = useState<File>();
	const [external, setExternal] = useState({ name: '', endpoint: '', family: 'generic' });
  const [cloud, setCloud] = useState<{
    provider: 'openai' | 'xai' | 'gemini';
    name: string;
    remoteId: string;
    modalities: string;
  }>({ provider: 'openai', name: '', remoteId: '', modalities: 'text' });
  const scanCatalog = async () => {
    if (!catalogDirectory.trim())
      return;
    setCatalogBusy(true);
    try {
      setCatalogEntries(await api<LocalCatalogEntry[]>(`models/catalog?directory=${encodeURIComponent(catalogDirectory.trim())}`));
    } catch (e) {
      fail((e as Error).message);
    } finally {
      setCatalogBusy(false);
    }
  };
  useEffect(() => {
    if (!catalogDirectory) {
      const localBase = baseModels.find(model => !model.provider);
      if (localBase)
        setCatalogDirectory(localBase.path);
    }
  }, [catalogDirectory, models]);
  useEffect(() => { void scanCatalog(); }, [catalogDirectory, models.length]);
  const registerCatalogEntry = async (entry: LocalCatalogEntry) => {
    try {
      await api('models', { method: 'POST', body: JSON.stringify({ name: entry.name, path: entry.path, format: entry.format, family: entry.family, modalities: ['text'], kind: 'base' }) });
      done(`${entry.name} registered. Start it from the Base models catalog when ready.`);
    } catch (e) {
      fail((e as Error).message);
    }
  };
  const addLocal = async (event: FormEvent) => {
    event.preventDefault();
    try {
      await api('models', { method: 'POST', body: JSON.stringify({ name: local.name, path: local.path, format: 'safetensors', family: local.family, modalities: ['text'], kind: 'base' }) });
      setLocal({ name: '', path: '', family: 'qwen' });
      done('Local model registered. Start it from the Base models catalog when ready.');
    } catch (e) {
      fail((e as Error).message);
    }
  };
  const submit = async (event: FormEvent) => { event.preventDefault(); if (!file)
    return fail('Choose a model file.'); try {
    const data = new FormData();
    data.append('file', file);
    data.append('name', name || file.name);
    data.append('kind', kind);
    data.append('endpoint', endpoint);
    data.append('family', family);
    data.append('modalities', modalities);
    await api('models/upload', { method: 'POST', body: data });
    setName('');
    setFile(undefined);
    done('Model uploaded.');
  }
  catch (e) {
    fail((e as Error).message);
  } };
  const addCloud = async (event: FormEvent) => { event.preventDefault(); try {
    await api('models', { method: 'POST', body: JSON.stringify({ name: cloud.name, kind: 'base', provider: cloud.provider, remoteId: cloud.remoteId, modalities: cloud.modalities.split(',') }) });
    setCloud(current => ({ ...current, name: '', remoteId: '' }));
    done('Cloud model added. Its API key stays in .env.');
  }
  catch (e) {
    fail((e as Error).message);
  } };
	const addExternal = async (event: FormEvent) => {
		event.preventDefault();
		try {
			await api('models', { method: 'POST', body: JSON.stringify({ name: external.name, endpoint: external.endpoint, family: external.family, format: 'api', modalities: ['text'], kind: 'base' }) });
			setExternal({ name: '', endpoint: '', family: 'generic' });
			done('Local endpoint registered. It is ready once its runner is healthy.');
		} catch (error) {
			fail((error as Error).message);
		}
	};
  return <div className="grid">
    <Card title="Model catalog">
      <p className="hint">Scan a trusted local directory to catalog downloaded checkpoints. Sypra detects SafeTensors directories with <code>config.json</code> plus <code>.gguf</code> and <code>.onnx</code> model files.</p>
      <form className="catalog-directory" onSubmit={event => { event.preventDefault(); void scanCatalog(); }}><label>Downloaded-model directory<input value={catalogDirectory} onChange={event => setCatalogDirectory(event.target.value)} placeholder="C:\models" /></label><button disabled={!catalogDirectory.trim() || catalogBusy}>{catalogBusy ? 'Scanning…' : 'Scan directory'}</button></form>
      {catalogEntries.length ? <div className="model-catalog">{catalogEntries.map(entry => <article key={entry.path} className="model-catalog-item"><div><h3>{entry.name}</h3><p>{entry.family} · {entry.format} · {entry.path}</p></div><div className="model-catalog-meta"><span className={entry.registered ? 'model-status production' : 'model-status archived'}>{entry.registered ? 'Registered' : 'Downloaded'}</span>{!entry.registered && <button type="button" className="make-default" onClick={() => void registerCatalogEntry(entry)}>Register</button>}</div></article>)}</div> : <p className="empty">{catalogDirectory ? 'No model directories found here.' : 'Choose the parent folder that contains your downloaded Qwen directory.'}</p>}
    </Card>
    <Card title="Register a downloaded model">
      <p className="hint">Register a SafeTensors model directory here. Native Start runs Qwen directly; GGUF and ONNX files can be registered from the catalogue and started after their managed local runtime is configured.</p>
      <form onSubmit={addLocal}>
        <label>Display name<input required value={local.name} onChange={event => setLocal({ ...local, name: event.target.value })} placeholder="e.g. Qwen3 0.6B" /></label>
        <label>Model directory<input required value={local.path} onChange={event => setLocal({ ...local, path: event.target.value })} placeholder="C:\models\Qwen3-0.6B" /></label>
        <label>Model family<select value={local.family} onChange={event => setLocal({ ...local, family: event.target.value })}><option value="qwen">Qwen</option><option value="llama">Llama</option><option value="mistral">Mistral / Mixtral</option><option value="gemma">Gemma</option><option value="phi">Phi</option><option value="generic">Generic</option></select></label>
        <button>Register local model</button>
      </form>
    </Card>
    <Card title="Connect a local model endpoint">
      <p className="hint">Use this for an independently started OpenAI-compatible local model process. Sypra does not own or stop external endpoints.</p>
      <form onSubmit={addExternal}>
        <label>Display name<input required value={external.name} onChange={event => setExternal({ ...external, name: event.target.value })} placeholder="Local test model" /></label>
        <label>Endpoint<input required value={external.endpoint} onChange={event => setExternal({ ...external, endpoint: event.target.value })} placeholder="http://127.0.0.1:8082/v1" /></label>
        <label>Model family<select value={external.family} onChange={event => setExternal({ ...external, family: event.target.value })}><option value="generic">Generic OpenAI-compatible</option><option value="qwen">Qwen</option><option value="llama">Llama</option><option value="mistral">Mistral / Mixtral</option><option value="gemma">Gemma</option><option value="phi">Phi</option></select></label>
        <button>Connect local endpoint</button>
      </form>
    </Card>
    <Card title="Base models">
      <p className="hint">The default base model is selected automatically for every new chat. Fine-tuned models remain explicit selections because they carry their own adapter and parent-model binding.</p>
      {baseModels.length ? <div className="model-catalog">{baseModels.map(model => <article key={model.id} className="model-catalog-item"><div><h3>{model.name}</h3><p>{model.provider ? `${model.provider} cloud model` : model.format || 'Local model'}{model.family ? ` · ${model.family}` : ''}</p>{runnerFor(model.id)?.error && <p className="runner-error">{runnerFor(model.id)?.error}</p>}</div><div className="model-catalog-meta">{model.default ? <span className="model-status production">Default</span> : <button type="button" className="make-default" onClick={() => void setDefaultBaseModel(model.id)}>Make default</button>}{runnerControl(model)}</div></article>)}</div> : <p className="empty">No base models yet. The first base model you add becomes the workspace default.</p>}
    </Card>
    <Card title="Fine-tuned models">
      <p className="hint">Fine-tuned models are deployable SLMs. Each is shown with its parent model, adapter method, deployment status, and validation result.</p>
      {fineTunedModels.length ? <div className="model-catalog">{fineTunedModels.map(model => <article key={model.id} className="model-catalog-item"><div><h3>{model.name}</h3><p>{model.parentModel || 'Parent model unavailable'} · {model.method?.toUpperCase() || 'Adapter'} · {model.format || 'adapter'}</p>{runnerFor(model.id)?.error && <p className="runner-error">{runnerFor(model.id)?.error}</p>}</div><div className="model-catalog-meta"><span className={`model-status ${model.status || 'candidate'}`}>{model.status || 'candidate'}</span><span>{model.metrics?.validationLoss !== undefined ? `Validation loss ${model.metrics.validationLoss.toFixed(4)}` : 'No validation metric recorded'}</span><span>{model.version ? `Version ${model.version}` : 'Unversioned'}</span>{runnerControl(model)}</div></article>)}</div> : <p className="empty">No fine-tuned models yet. Completed LoRA or QLoRA runs appear here as deployable SLMs.</p>}
    </Card>
    <Card title="Upload a local model">
      <p className="hint">Upload a model artifact when needed. The built-in native runner currently executes Qwen SafeTensors checkpoints; GGUF and ONNX artifacts are catalogued but are not delegated to third-party engines.</p>
      <form onSubmit={submit}>
        <label>Model role<select value={kind} onChange={e => setKind(e.target.value as 'base' | 'slm')}><option value="base">Base model</option><option value="slm">Fine-tuned SLM</option></select></label>
        <label>Display name<input value={name} onChange={e => setName(e.target.value)}/></label>
        <label>Model file<input required type="file" onChange={e => setFile(e.target.files?.[0])}/></label>
        <label>Runner endpoint<input value={endpoint} onChange={e => setEndpoint(e.target.value)} placeholder="http://127.0.0.1:8081/v1"/></label>
        <label>Model family<select value={family} onChange={e => setFamily(e.target.value)}><option value="generic">Generic OpenAI-compatible</option><option value="qwen">Qwen</option><option value="llama">Llama</option><option value="mistral">Mistral / Mixtral</option><option value="gemma">Gemma</option><option value="phi">Phi</option></select></label>
        <label>Capabilities<input value={modalities} onChange={e => setModalities(e.target.value)} placeholder="text,image,audio,video,file"/></label>
        <button>Upload model</button>
      </form>
    </Card>
    <Card title="Connect a cloud model">
      <p className="hint">The key is read only from the API server&apos;s .env file, never from this UI. Cloud models can use local RAG; native fine-tuning remains local-Qwen only.</p>
      <form onSubmit={addCloud}>
        <label>Provider<select value={cloud.provider} onChange={e => setCloud({ ...cloud, provider: e.target.value as typeof cloud.provider })}><option value="openai">OpenAI</option><option value="xai">xAI Grok</option><option value="gemini">Google Gemini</option></select></label>
        <label>Display name<input value={cloud.name} onChange={e => setCloud({ ...cloud, name: e.target.value })} placeholder="Optional"/></label>
        <label>Provider model ID<input required value={cloud.remoteId} onChange={e => setCloud({ ...cloud, remoteId: e.target.value })} placeholder={cloud.provider === 'xai' ? 'grok-4.7' : cloud.provider === 'gemini' ? 'gemini-2.5-pro' : 'Your OpenAI model ID'}/></label>
        <label>Capabilities<input value={cloud.modalities} onChange={e => setCloud({ ...cloud, modalities: e.target.value })} placeholder="text,image,audio,video,file"/></label>
        <button>Connect cloud model</button>
      </form>
    </Card>
    <Card title="Registered model details"><Table rows={models} cols={['name', 'family', 'modalities', 'provider', 'remoteId', 'kind', 'parentModel', 'method', 'status', 'version', 'format', 'path', 'endpoint']}/></Card>
  </div>;
}
function Card({ title, children }: {
  title: string;
  children: React.ReactNode;
}) { return <section className="card"><h2>{title}</h2>{children}</section>; }
function Table({ rows, cols }: {
  rows: any[];
  cols: string[];
}) { return rows.length ? <div className="table">{rows.map((row, index) => <div className="row" key={row.id || index}>{cols.map(column => <span key={column}><small>{column}</small>{typeof row[column] === 'number' && column === 'progress' ? `${Math.round(row[column] * 100)}%` : String(row[column] ?? '—')}</span>)}</div>)}</div> : <p className="empty">Nothing here yet.</p>; }
createRoot(document.getElementById('root')!).render(<App />);
