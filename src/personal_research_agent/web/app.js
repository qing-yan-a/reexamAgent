const { createApp, nextTick } = Vue;

const reexamApp = createApp({
  data() {
    return {
      // sessions 是 /sessions 的平铺列表；folders 是 /folders 的工作区树。
      sessions: [],
      folders: [],
      selectedFolder: "",
      openFolders: {},
      // currentSessionId 决定当前 WebSocket 连接到哪个 LangGraph thread。
      currentSessionId: "",
      socket: null,
      connected: false,
      // messages 只负责当前浏览器页面展示；刷新后由后端 history_reset 重新填充。
      messages: [],
      draftMessage: "",
      interrupt: null,
      // panel 是右侧研究面板的数据，来自 /research-panel 或 WebSocket 更新。
      panel: {},
      rightTab: "panel",
      newSession: { school: "", major: "", year: "latest" },
      selectedDraft: "",
      draftContent: "",
      saveStatus: "",
      selectedSourceKeys: [],
      openToolMessages: {},
      nextUiMessageId: 1,
    };
  },
  computed: {
    filteredSessions() {
      // 当前选中文件夹下的 sessions 已经由后端挂在 folder.sessions 上。
      const folder = this.folders.find((item) => item.output_dir === this.selectedFolder);
      return folder?.sessions || [];
    },
    connectionLabel() {
      // 顶部栏显示当前 session 的 WebSocket 连接状态。
      if (!this.currentSessionId) return "请选择或创建 session";
      return this.connected ? `已连接 ${this.currentSessionId}` : `未连接 ${this.currentSessionId}`;
    },
    interruptTitle() {
      // interrupt 可能是工具审批、进入复试流程确认，也可能是搜索循环的下一步决策。
      if (!this.interrupt) return "";
      if (this.interrupt.type === "approval_required") return "工具审批";
      if (this.interrupt.type === "reexam_route_confirmation_required") return "进入复试搜索流程？";
      return "复试搜索决策";
    },
    canSend() {
      // interrupt 等待用户决策时，先禁止继续发送普通聊天消息。
      return this.connected && this.draftMessage.length > 0 && !this.interrupt;
    },
  },
  async mounted() {
    // 页面首次加载时拉取 session 和工作区树，再默认连接最新 session。
    await this.refreshAll();
    if (this.sessions.length) {
      this.selectSession(this.sessions[0].session_id);
    }
  },
  methods: {
    async refreshAll() {
      // 同时刷新左侧 session 列表和工作区树。
      const [sessionRes, folderRes] = await Promise.all([fetch("/sessions"), fetch("/folders")]);
      this.sessions = (await sessionRes.json()).sessions || [];
      this.folders = (await folderRes.json()).folders || [];
      const folderIds = new Set(this.folders.map((item) => item.output_dir));
      if ((!this.selectedFolder || !folderIds.has(this.selectedFolder)) && this.folders.length) {
        this.selectedFolder = this.folders[0].output_dir;
      }
      if (this.selectedFolder) {
        this.openFolder(this.selectedFolder);
      }
      this.openFolderForSession(this.currentSessionId);
      if (this.currentSessionId) {
        await this.refreshPanel();
      }
    },
    async createSession() {
      // 创建 session 后重新刷新工作区树，并切换到新 session。
      const res = await fetch("/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(this.newSession),
      });
      if (!res.ok) {
        alert(await res.text());
        return;
      }
      const data = await res.json();
      await this.refreshAll();
      this.selectSession(data.session.session_id);
    },
    selectSession(sessionId) {
      // 切换 session 会重建 WebSocket；历史消息会由后端 history_reset 重新发送。
      this.currentSessionId = sessionId;
      this.openFolderForSession(sessionId);
      this.messages = [];
      this.interrupt = null;
      this.selectedDraft = "";
      this.draftContent = "";
      this.connectSocket();
      this.refreshPanel();
    },
    async deleteSession(sessionId) {
      // 删除 session 只删除 memory/sessions 下的状态，不删除 test/ 下的资料文件夹。
      if (!confirm(`确认删除 session：${sessionId}？资料输出目录不会删除。`)) return;
      const res = await fetch(`/sessions/${sessionId}`, { method: "DELETE" });
      if (!res.ok) {
        alert(await res.text());
        return;
      }
      if (this.currentSessionId === sessionId) {
        if (this.socket) this.socket.close();
        this.currentSessionId = "";
        this.panel = {};
        this.messages = [];
      }
      await this.refreshAll();
      if (!this.currentSessionId && this.sessions.length) {
        this.selectSession(this.sessions[0].session_id);
      }
    },
    openFolder(outputDir) {
      // 用对象保存展开状态，避免刷新 folders 数组时丢失每个文件夹的开合状态。
      if (!outputDir) return;
      this.openFolders = { ...this.openFolders, [outputDir]: true };
    },
    toggleFolder(outputDir) {
      // 点击文件夹行时切换展开状态，并把它设为当前选中文件夹。
      this.selectedFolder = outputDir;
      this.openFolders = { ...this.openFolders, [outputDir]: !this.openFolders[outputDir] };
    },
    isFolderOpen(outputDir) {
      return Boolean(this.openFolders[outputDir]);
    },
    openFolderForSession(sessionId) {
      // 选中 session 时自动展开它所属的工作区文件夹。
      if (!sessionId) return;
      const folder = this.folders.find((item) =>
        (item.sessions || []).some((session) => session.session_id === sessionId)
      );
      if (!folder) return;
      this.selectedFolder = folder.output_dir;
      this.openFolder(folder.output_dir);
    },
    connectSocket() {
      // 每个 session 对应一个 WebSocket 连接，也对应后端的 thread_id。
      if (this.socket) this.socket.close();
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      this.socket = new WebSocket(`${protocol}://${location.host}/ws/sessions/${this.currentSessionId}`);
      this.socket.onopen = () => {
        this.connected = true;
      };
      this.socket.onclose = () => {
        this.connected = false;
      };
      this.socket.onerror = () => {
        this.connected = false;
      };
      this.socket.onmessage = (event) => this.handleSocketEvent(JSON.parse(event.data));
    },
    handleSocketEvent(event) {
      // 后端连接成功后会先发 history_reset，用 checkpoint 恢复聊天历史。
      if (event.type === "history_reset") {
        this.messages = this.normalizeMessages(event.messages || []);
        this.openToolMessages = {};
        this.scrollMessages();
        return;
      }
      // assistant 的流式回答从 message_start 开始创建气泡。
      if (event.type === "message_start") {
        if (event.role === "assistant" || event.role === "tool") {
          this.messages.push(this.createMessage(event.role, "", event.message_id));
        }
        this.scrollMessages();
        return;
      }
      // message_delta 是流式 token；按 message_id 追加到对应的 Agent 气泡里。
      if (event.type === "message_delta") {
        const target = event.message_id
          ? this.findMessageByServerId(event.message_id)
          : this.messages[this.messages.length - 1];
        if (target && (target.role === "assistant" || target.role === "tool")) {
          target.content += event.delta || "";
        }
        this.scrollMessages();
        return;
      }
      // interrupt 事件会让页面显示审批按钮、流程确认按钮或复试搜索决策按钮。
      if (
        event.type === "approval_required" ||
        event.type === "reexam_route_confirmation_required" ||
        event.type === "reexam_decision_required"
      ) {
        this.interrupt = event;
        return;
      }
      // 图执行后端可能更新 research_session，所以这里刷新右侧研究面板。
      if (event.type === "research_panel_update") {
        this.panel = event.panel || {};
        this.syncSelectedSourceKeys();
        return;
      }
      // 后端异常不会直接弹窗，而是显示成一条 Agent 错误消息。
      if (event.type === "error") {
        this.messages.push(this.createMessage("assistant", `错误：${event.message}`));
        this.scrollMessages();
      }
    },
    sendMessage() {
      // 先乐观显示用户消息，再把内容通过 WebSocket 发给后端图执行。
      if (!this.canSend) return;
      const message = this.draftMessage;
      this.messages.push(this.createMessage("user", message));
      this.socket.send(JSON.stringify({ type: "user_message", message }));
      this.draftMessage = "";
      this.scrollMessages();
    },
    sendDecision(type, value) {
      // 把 interrupt 的用户选择回传给后端，后端会用 Command(resume=...) 继续图执行。
      if (!this.socket || !this.interrupt) return;
      this.messages.push(this.createMessage("user", `选择：${value}`));
      this.socket.send(JSON.stringify({ type, value }));
      this.interrupt = null;
      this.scrollMessages();
    },
    async refreshPanel() {
      // 右侧研究面板也支持独立 HTTP 刷新，不依赖聊天流。
      if (!this.currentSessionId) return;
      const res = await fetch(`/sessions/${this.currentSessionId}/research-panel`);
      if (res.ok) {
        this.panel = (await res.json()).panel || {};
        this.syncSelectedSourceKeys();
      }
    },
    syncSelectedSourceKeys() {
      // 让候选来源列表里的 checkbox 和 session.selected_sources 保持同步。
      const keys = new Set((this.panel.selected_sources || []).map((item) => item.source_key).filter(Boolean));
      const visibleKeys = new Set((this.panel.candidate_sources || []).map((item) => item.source_key).filter(Boolean));
      this.selectedSourceKeys = Array.from(keys).filter((key) => visibleKeys.has(key));
    },
    isSelectedSource(sourceKey) {
      // 候选来源卡片右侧显示“已保留”的判断。
      return (this.panel.selected_sources || []).some((item) => item.source_key === sourceKey);
    },
    async saveSelectedSources() {
      // 保存前端勾选的候选来源，后端会写入 research_session.selected_sources。
      if (!this.currentSessionId) return;
      const res = await fetch(`/sessions/${this.currentSessionId}/selected-sources`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_keys: this.selectedSourceKeys }),
      });
      if (!res.ok) {
        alert(await res.text());
        return;
      }
      const data = await res.json();
      this.panel = data.panel || this.panel;
      this.syncSelectedSourceKeys();
      this.messages.push(
        this.createMessage("assistant", `已保留 ${data.selected_count || 0} 个来源。后续可以要求我抽取这些来源正文。`)
      );
      this.scrollMessages();
    },
    async loadDraft() {
      // 从后端读取当前 output_dir 下的 Markdown 草稿。
      if (!this.selectedDraft) return;
      const res = await fetch(`/sessions/${this.currentSessionId}/drafts/${encodeURIComponent(this.selectedDraft)}`);
      if (res.ok) {
        this.draftContent = (await res.json()).content || "";
      }
    },
    async saveDraft() {
      // 保存 Markdown 草稿；后端会校验文件名和路径，防止写出 output_dir。
      if (!this.selectedDraft) return;
      const res = await fetch(`/sessions/${this.currentSessionId}/drafts/${encodeURIComponent(this.selectedDraft)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: this.draftContent }),
      });
      if (res.ok) {
        const data = await res.json();
        this.saveStatus = `已保存：${data.path}`;
        await this.refreshPanel();
      } else {
        this.saveStatus = await res.text();
      }
    },
    scrollMessages() {
      // 每次消息变化后滚动到底部，保持最新对话可见。
      nextTick(() => {
        const el = this.$refs.messageList;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },
    scrollToolDetailIntoView(message) {
      nextTick(() => {
        const list = this.$refs.messageList;
        if (!list || !message) return;
        const key = this.toolMessageKey(message);
        const detail = list.querySelector(`[data-message-key="${key}"] .tool-detail-text`);
        if (detail) detail.scrollIntoView({ block: "nearest" });
      });
    },
    nextMessageUiId() {
      const value = `ui_${this.nextUiMessageId}`;
      this.nextUiMessageId += 1;
      return value;
    },
    createMessage(role, content = "", serverId = "") {
      // 后端 LangChain message.id 可能重复或为空；前端渲染和展开状态使用 uiId 保持一条气泡一个身份。
      const uiId = this.nextMessageUiId();
      return {
        id: serverId || uiId,
        serverId: serverId || "",
        uiId,
        role,
        content,
      };
    },
    normalizeMessages(messages) {
      // history_reset 来自 checkpoint，不能假设每条 LangChain message.id 都适合当 Vue key。
      return messages.map((message) => {
        const serverId = String(message?.serverId || message?.id || "");
        const uiId = message?.uiId || this.nextMessageUiId();
        return {
          ...message,
          id: serverId || uiId,
          serverId,
          uiId,
          content: String(message?.content || ""),
        };
      });
    },
    findMessageByServerId(serverId) {
      // 从尾部找，避免后端重复 message_id 时把新 delta 追加到旧气泡。
      const key = String(serverId || "");
      for (let index = this.messages.length - 1; index >= 0; index -= 1) {
        const message = this.messages[index];
        if ((message.serverId || message.id) === key) return message;
      }
      return null;
    },
    isToolLikeMessage(message) {
      // ToolMessage 和“模型请求工具调用”都默认折叠，避免大段 JSON 把聊天窗口撑爆。
      const content = message?.content || "";
      return message?.role === "tool" || content.startsWith("模型请求工具调用：");
    },
    toolMessageSummary(message) {
      // 折叠状态只露出工具名；展开后再看完整入参/返回值。
      const content = String(message?.content || "").trim();
      if (message?.role === "tool") {
        const parsed = this.parseToolMessageContent(content);
        return `工具结果：${parsed.toolName}${this.toolPayloadSummary(parsed)}`;
      }
      const callLine = content.split(/\r?\n/).find((line) => line.includes("(")) || "工具调用";
      const toolName = callLine.split("(")[0].trim();
      return `工具调用：${toolName || "待执行工具"}`;
    },
    toolMessageDetail(message) {
      // 展开后展示完整工具调用/结果；如果内容没推送到前端，也给出明确提示方便排查。
      const content = String(message?.content || "").trim();
      if (!content) return "当前工具消息没有可展开内容。";
      if (message?.role === "tool") {
        return this.formatToolMessageDetail(this.parseToolMessageContent(content));
      }
      return content;
    },
    parseToolMessageContent(content) {
      const lines = String(content || "").trim().split(/\r?\n/);
      const firstLine = (lines.shift() || "tool").trim() || "tool";
      const payloadText = lines.join("\n").trim();
      let payload = null;
      if (payloadText) {
        try {
          payload = JSON.parse(payloadText);
        } catch (_error) {
          payload = null;
        }
      }
      return { toolName: firstLine, payloadText, payload };
    },
    toolPayloadSummary(parsed) {
      const payload = parsed?.payload;
      if (!payload || typeof payload !== "object") return "";
      if (parsed.toolName === "web_search") {
        return ` · ${payload.result_count ?? 0} 条结果 · ${payload.query || ""}`;
      }
      if (parsed.toolName === "source_review") {
        return ` · 初筛 ${payload.review_count ?? 0} 条`;
      }
      if (parsed.toolName === "evaluate_research_readiness") {
        const gaps = Array.isArray(payload.open_gaps) ? payload.open_gaps.length : 0;
        return ` · ${payload.draft_ready ? "草稿可生成" : "草稿未就绪"} · ${gaps} 个缺口`;
      }
      if (payload.message) return ` · ${payload.message}`;
      return "";
    },
    formatToolMessageDetail(parsed) {
      const payload = parsed.payload;
      if (!payload || typeof payload !== "object") {
        return [`工具：${parsed.toolName}`, parsed.payloadText || "工具没有返回结构化内容。"].join("\n\n");
      }
      if (parsed.toolName === "web_search") {
        const lines = [
          `工具：${parsed.toolName}`,
          `query：${payload.query || "-"}`,
          `结果数：${payload.result_count ?? 0}`,
        ];
        for (const item of payload.results || []) {
          lines.push("");
          lines.push(`[${item.source_index ?? "-"}] ${item.title || "(无标题)"}`);
          lines.push(`URL：${item.url || "-"}`);
          lines.push(`来源：${item.source || "-"}`);
          if (item.snippet) lines.push(`摘要：${item.snippet}`);
        }
        return lines.join("\n");
      }
      if (parsed.toolName === "source_review") {
        const lines = [
          `工具：${parsed.toolName}`,
          `研究目标：${payload.research_goal || "-"}`,
          `初筛数：${payload.review_count ?? 0}`,
        ];
        for (const item of payload.reviews || []) {
          lines.push("");
          lines.push(`[${item.source_index ?? "-"}] ${item.title || "(无标题)"}`);
          lines.push(`URL：${item.url || "-"}`);
          lines.push(`相关性：${item.relevance || "-"} / 可信度：${item.credibility_hint || "-"}`);
          lines.push(`建议：${item.next_action || "-"}`);
          if (item.review_note) lines.push(`备注：${item.review_note}`);
        }
        return lines.join("\n");
      }
      if (parsed.toolName === "evaluate_research_readiness") {
        const lines = [
          `工具：${parsed.toolName}`,
          `草稿状态：${payload.draft_ready ? "可生成" : "未就绪"}`,
          `说明：${payload.readiness_note || "-"}`,
        ];
        const gaps = Array.isArray(payload.open_gaps) ? payload.open_gaps : [];
        if (gaps.length) {
          lines.push("");
          lines.push("资料缺口：");
          lines.push(...gaps.map((gap) => `- ${gap}`));
        }
        lines.push("");
        lines.push("完整 JSON：");
        lines.push(JSON.stringify(payload, null, 2));
        return lines.join("\n");
      }
      return [`工具：${parsed.toolName}`, "完整 JSON：", JSON.stringify(payload, null, 2)].join("\n");
    },
    toolMessageKey(message) {
      // 展开状态存在组件顶层 map 里；key 使用前端 uiId，避免后端 message.id 重复时串状态。
      return String(message?.uiId || message?.id || `${message?.role || "message"}:${this.messages.indexOf(message)}`);
    },
    isToolMessageOpen(message) {
      return Boolean(this.openToolMessages[this.toolMessageKey(message)]);
    },
    toggleToolMessage(message) {
      // 工具消息默认折叠；点击摘要行再展开完整参数或返回 JSON。
      if (!message) return;
      const key = this.toolMessageKey(message);
      const nextOpen = !this.openToolMessages[key];
      this.openToolMessages = {
        ...this.openToolMessages,
        [key]: nextOpen,
      };
      if (nextOpen) this.scrollToolDetailIntoView(message);
    },
    toolDebugRows() {
      // Console 调试用：window.__reexamApp.toolDebugRows()
      return this.messages.slice(-8).map((message) => ({
        key: this.toolMessageKey(message),
        serverId: message.serverId || message.id || "",
        role: message.role,
        isTool: this.isToolLikeMessage(message),
        len: (message.content || "").length,
        open: this.isToolMessageOpen(message),
        summary: this.isToolLikeMessage(message) ? this.toolMessageSummary(message) : "",
        detailHead: this.isToolLikeMessage(message) ? this.toolMessageDetail(message).slice(0, 80) : "",
      }));
    },
    escapeHtml(value) {
      // 先转义，再做轻量 Markdown 替换，避免把模型输出当成原生 HTML 执行。
      return String(value || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    },
    renderInlineMarkdown(text) {
      // 只支持聊天里最常见的粗体、行内代码和链接，保持实现可控。
      return this.escapeHtml(text)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    },
    renderMarkdown(value) {
      // 轻量 Markdown 渲染：把列表、段落、粗体处理成 HTML，解决界面里满屏星号的问题。
      const lines = String(value || "").replace(/\r\n/g, "\n").split("\n");
      const blocks = [];
      let paragraph = [];
      let listItems = [];
      let orderedItems = [];

      const flushParagraph = () => {
        if (!paragraph.length) return;
        blocks.push(`<p>${paragraph.map((line) => this.renderInlineMarkdown(line)).join("<br>")}</p>`);
        paragraph = [];
      };
      const flushList = () => {
        if (listItems.length) {
          blocks.push(`<ul>${listItems.map((item) => `<li>${this.renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
          listItems = [];
        }
        if (orderedItems.length) {
          blocks.push(`<ol>${orderedItems.map((item) => `<li>${this.renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
          orderedItems = [];
        }
      };

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (!line.trim()) {
          flushParagraph();
          flushList();
          continue;
        }
        const heading = line.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
          flushParagraph();
          flushList();
          const level = Math.min(heading[1].length + 2, 5);
          blocks.push(`<h${level}>${this.renderInlineMarkdown(heading[2])}</h${level}>`);
          continue;
        }
        const unordered = line.match(/^[-*]\s+(.+)$/);
        if (unordered) {
          flushParagraph();
          if (orderedItems.length) flushList();
          listItems.push(unordered[1]);
          continue;
        }
        const ordered = line.match(/^\d+[.)]\s+(.+)$/);
        if (ordered) {
          flushParagraph();
          if (listItems.length) flushList();
          orderedItems.push(ordered[1]);
          continue;
        }
        flushList();
        paragraph.push(line);
      }
      flushParagraph();
      flushList();
      return blocks.join("");
    },
    messageRoleLabel(role) {
      // 前端显示用角色名；tool 消息来自 LangGraph 工具执行结果。
      if (role === "user") return "你";
      if (role === "tool") return "工具";
      return "Agent";
    },
  },
}).mount("#app");

// 暴露给浏览器 Console 的调试入口，便于排查 WebSocket 消息和前端渲染状态。
window.__reexamApp = reexamApp;
