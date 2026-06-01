import React, { useState, useEffect, useRef } from "react";
import {
  Scale,
  Search,
  Plus,
  Trash2,
  Send,
  Sparkles,
  Headphones,
  MessageSquare,
  Info,
  Clock,
  Compass,
  ArrowRight,
  User,
  Loader2,
  ExternalLink
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  sources?: any[];
  intent?: string;
  warning?: string;
  latency_ms?: number;
  is_pro?: boolean;
}

interface ChatSession {
  session_id: string;
  title: string;
  created_at: number;
  updated_at: number;
}

// Câu hỏi gợi ý ban đầu
const SUGGESTIONS = [
  {
    text: "Nâng chuẩn giáo viên mầm non thực hiện theo lộ trình nào?",
    desc: "Chi tiết lộ trình, đối tượng & chính sách hỗ trợ theo Nghị định 71/2020."
  },
  {
    text: "Sinh viên sư phạm nghỉ học tạm thời có bị cắt hỗ trợ sinh hoạt phí không?",
    desc: "Quy định về chi trả học phí và sinh hoạt phí theo Nghị định 116/2020."
  },
  {
    text: "Điều kiện thăng hạng từ giáo viên tiểu học Hạng III lên Hạng II là gì?",
    desc: "Tiêu chuẩn về bằng cấp, chứng chỉ và số năm giữ hạng theo Thông tư 02 & 08."
  },
  {
    text: "Quy định về kiểm tra bù và đánh giá lại học sinh trung học ra sao?",
    desc: "Căn cứ pháp lý theo Thông tư 22/2021 về đánh giá học sinh."
  }
];

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("default");
  const [messages, setMessages] = useState<Message[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  // RAG Pipeline Real-time status
  const [pipelineStatus, setPipelineStatus] = useState("");
  const [activeIntent, setActiveIntent] = useState("");

  // Highlighted citation index
  const [hoveredCitationIdx, setHoveredCitationIdx] = useState<number | null>(null);

  // Trạng thái hiển thị modal hỗ trợ
  const [showSupportModal, setShowSupportModal] = useState(false);

  // Trạng thái phiên bản Pro
  const [isProMode, setIsProMode] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load danh sách lịch sử khi khởi chạy
  useEffect(() => {
    fetchSessions();
  }, []);

  // Tự động cuộn xuống cuối khi có tin nhắn mới
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pipelineStatus]);

  // Load tin nhắn của session đang hoạt động
  useEffect(() => {
    if (activeSessionId) {
      loadSessionDetail(activeSessionId);
    }
  }, [activeSessionId]);

  const fetchSessions = async () => {
    try {
      const res = await fetch("/api/history");
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
      }
    } catch (e) {
      console.error("Lỗi lấy lịch sử chat:", e);
    }
  };

  const loadSessionDetail = async (sid: string) => {
    if (sid === "default") {
      setMessages([]);
      return;
    }
    try {
      const res = await fetch(`/api/history/${sid}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      }
    } catch (e) {
      console.error("Lỗi lấy chi tiết cuộc trò chuyện:", e);
    }
  };

  const handleCreateNewChat = () => {
    const newSid = Math.random().toString(36).substring(2, 10);
    setActiveSessionId(newSid);
    setMessages([]);
    setPipelineStatus("");
    setActiveIntent("");
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const handleDeleteSession = async (e: React.MouseEvent, sid: string) => {
    e.stopPropagation();
    if (!confirm("Bạn có chắc chắn muốn xóa cuộc hội thoại này?")) return;
    try {
      const res = await fetch(`/api/history/${sid}`, { method: "DELETE" });
      if (res.ok) {
        await fetchSessions();
        if (activeSessionId === sid) {
          setActiveSessionId("default");
          setMessages([]);
        }
      }
    } catch (err) {
      console.error("Lỗi xóa cuộc hội thoại:", err);
    }
  };

  // Gửi tin nhắn và xử lý SSE Stream
  const handleSendMessage = async (textToSend?: string) => {
    const text = (textToSend || inputValue).trim();
    if (!text || isStreaming) return;

    setInputValue("");
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }

    // Nếu đang ở session mặc định chưa lưu, hãy tạo ID ngẫu nhiên để persist
    let currentSid = activeSessionId;
    if (activeSessionId === "default") {
      currentSid = Math.random().toString(36).substring(2, 10);
      setActiveSessionId(currentSid);
    }

    // Thêm câu hỏi của user vào danh sách tin nhắn ngay lập tức
    const userMsg: Message = {
      role: "user",
      content: text,
      timestamp: Date.now() / 1000
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);
    setPipelineStatus("Đang khởi tạo kết nối...");
    setActiveIntent("");

    // Khởi tạo tin nhắn trống cho AI để stream đổ vào
    const aiPlaceholder: Message = {
      role: "assistant",
      content: "",
      timestamp: Date.now() / 1000,
      sources: [],
      is_pro: isProMode
    };
    setMessages((prev) => [...prev, aiPlaceholder]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question: text,
          session_id: currentSid,
          include_sources: true,
          stream: true,
          is_pro: isProMode
        })
      });

      if (!response.body) {
        throw new Error("Không có phản hồi từ máy chủ");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = "";

      let accumulatedAnswer = "";
      let finalSources: any[] = [];
      let finalIntent = "";
      let finalWarning = "";
      let finalLatency = 0;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          // Giữ dòng cuối trong buffer vì có thể nó chưa trọn vẹn
          buffer = lines.pop() || "";

          for (const line of lines) {
            const cleanLine = line.trim();
            if (cleanLine.startsWith("data: ")) {
              try {
                const jsonStr = cleanLine.substring(6).trim();
                const parsed = JSON.parse(jsonStr);

                if (parsed.type === "status") {
                  setPipelineStatus(parsed.content);
                } else if (parsed.type === "meta") {
                  finalIntent = parsed.intent;
                  setActiveIntent(parsed.intent);
                } else if (parsed.type === "sources") {
                  finalSources = parsed.sources;
                  setMessages((prev) => {
                    const next = [...prev];
                    if (next.length > 0) {
                      next[next.length - 1].sources = finalSources;
                    }
                    return next;
                  });
                } else if (parsed.type === "token") {
                  accumulatedAnswer += parsed.content;
                  setMessages((prev) => {
                    const next = [...prev];
                    if (next.length > 0) {
                      next[next.length - 1].content = accumulatedAnswer;
                    }
                    return next;
                  });
                } else if (parsed.type === "done") {
                  accumulatedAnswer = parsed.answer || accumulatedAnswer;
                  finalSources = parsed.sources || finalSources;
                  finalIntent = parsed.intent || finalIntent;
                  finalWarning = parsed.warning || "";
                  finalLatency = parsed.latency_ms || 0;
                  const finalIsPro = parsed.is_pro !== undefined ? parsed.is_pro : isProMode;

                  setMessages((prev) => {
                    const next = [...prev];
                    if (next.length > 0) {
                      next[next.length - 1] = {
                        role: "assistant",
                        content: accumulatedAnswer,
                        sources: finalSources,
                        intent: finalIntent,
                        warning: finalWarning,
                        latency_ms: finalLatency,
                        is_pro: finalIsPro,
                        timestamp: Date.now() / 1000
                      };
                    }
                    return next;
                  });

                  setPipelineStatus("");
                  done = true;
                }
              } catch (e) {
                // Ignore incomplete JSON chunks
              }
            }
          }
        }
      }
    } catch (err) {
      console.error("Lỗi gọi API chat:", err);
      setMessages((prev) => {
        const next = [...prev];
        if (next.length > 0) {
          next[next.length - 1].content = "Đã xảy ra lỗi kết nối với trợ lý AI. Vui lòng kiểm tra lại backend và thử lại!";
        }
        return next;
      });
      setPipelineStatus("");
    } finally {
      setIsStreaming(false);
      fetchSessions(); // Làm mới danh sách Sidebar
    }
  };

  // Trình render Markdown cao cấp hỗ trợ Bảng biểu, Tiêu đề, Danh sách và ngắt dòng khoa học
  const renderMessageContent = (content: string, sources: any[] = []) => {
    if (!content) return null;

    const lines = content.split("\n");
    const elements: React.ReactNode[] = [];

    let currentTableRows: string[][] = [];
    let inTable = false;

    let currentListItems: React.ReactNode[] = [];
    let inList = false;

    const flushTable = (key: string) => {
      if (currentTableRows.length === 0) return;

      let headers = currentTableRows[0];
      let rows = currentTableRows.slice(1);

      // Kiểm tra xem dòng thứ 2 có phải phân tách bảng |---|---| không
      if (rows.length > 0 && rows[0].every(cell => cell.trim().startsWith("-") || cell.trim() === "")) {
        rows = rows.slice(1);
      } else {
        headers = [];
        rows = currentTableRows;
      }

      elements.push(
        <div key={key} className="overflow-x-auto my-4.5 rounded-xl border border-slate-200 shadow-sm bg-white">
          <table className="min-w-full divide-y divide-slate-200 text-left">
            {headers.length > 0 && (
              <thead className="bg-slate-50 text-slate-700 font-bold text-xs uppercase tracking-wider">
                <tr>
                  {headers.map((h, i) => (
                    <th key={i} className="px-4 py-3 border-b border-slate-200 font-semibold">{h.trim()}</th>
                  ))}
                </tr>
              </thead>
            )}
            <tbody className="divide-y divide-slate-100 text-slate-600 bg-white">
              {rows.map((row, rIdx) => (
                <tr key={rIdx} className={rIdx % 2 === 0 ? "bg-white" : "bg-slate-50/30"}>
                  {row.map((cell, cIdx) => (
                    <td key={cIdx} className="px-4 py-3.5 text-[14.5px] leading-relaxed font-normal text-slate-700">
                      {parseInlineStyles(cell.trim(), sources)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      currentTableRows = [];
      inTable = false;
    };

    const flushList = (key: string) => {
      if (currentListItems.length === 0) return;
      elements.push(
        <ul key={key} className="list-disc pl-6 my-3.5 space-y-1.5 text-slate-700 leading-relaxed text-[16px]">
          {currentListItems}
        </ul>
      );
      currentListItems = [];
      inList = false;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      // 1. Phân tích cú pháp BẢNG BIỂU (Table)
      if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
        if (inList) flushList(`list-before-table-${i}`);

        inTable = true;
        const cells = line.split("|").slice(1, -1);
        currentTableRows.push(cells);
        continue;
      } else if (inTable) {
        flushTable(`table-${i}`);
      }

      // 2. Phân tích cú pháp DANH SÁCH (List Item)
      if (trimmed.startsWith("-") || trimmed.startsWith("*") || trimmed.startsWith("•")) {
        const text = trimmed.substring(1).trim();
        
        // Bỏ qua các list item rỗng hoặc chỉ chứa ký tự rác (--, -)
        if (text === "" || text === "--" || text === "-" || text.replace(/[-*]/g, "").trim() === "") {
            // Nếu list item là rác, ta không làm gì cả để loại bỏ nó
        } else {
            currentListItems.push(
              <li key={`li-${i}`} className="pl-1">
                {parseInlineStyles(text, sources)}
              </li>
            );
        }
        inList = true;
        continue;
      } else if (inList) {
        if (trimmed === "" || (!trimmed.startsWith("-") && !trimmed.startsWith("*") && !trimmed.startsWith("•"))) {
          flushList(`list-${i}`);
        }
      }

      // 3. Bỏ qua dòng trống nhưng ngắt đoạn
      if (trimmed === "") {
        continue;
      }

      // 4. Phân tích cú pháp TIÊU ĐỀ (Headings)
      if (trimmed.startsWith("#### ")) {
        elements.push(
          <h5 key={`h5-${i}`} className="text-[16px] font-bold text-blue-900 mt-4.5 mb-2 flex items-center gap-1.5">
            <span className="w-1.5 h-3.5 rounded bg-blue-500 inline-block"></span>
            {parseInlineStyles(trimmed.substring(5), sources)}
          </h5>
        );
        continue;
      }
      if (trimmed.startsWith("### ")) {
        elements.push(
          <h4 key={`h4-${i}`} className="text-lg font-bold text-blue-950 mt-5 mb-2.5">
            {parseInlineStyles(trimmed.substring(4), sources)}
          </h4>
        );
        continue;
      }
      if (trimmed.startsWith("## ")) {
        elements.push(
          <h3 key={`h3-${i}`} className="text-xl font-bold text-blue-950 mt-5.5 mb-3 border-b border-slate-100 pb-1.5">
            {parseInlineStyles(trimmed.substring(3), sources)}
          </h3>
        );
        continue;
      }

      // 5. Phân tích cú pháp ĐƯỜNG KẺ NGANG (Horizontal Rule)
      if (trimmed === "---") {
        elements.push(<hr key={`hr-${i}`} className="my-4 border-slate-200" />);
        continue;
      }

      // 6. ĐOẠN VĂN THÔNG THƯỜNG
      elements.push(
        <p key={`p-${i}`} className="text-slate-700 leading-relaxed mb-3 text-[16px] text-justify">
          {parseInlineStyles(trimmed, sources)}
        </p>
      );
    }

    // Ghi nhận các phần tử còn dư sau vòng lặp
    if (inTable) flushTable("table-end");
    if (inList) flushList("list-end");

    return elements;
  };

  // Nhận diện **bold**, các thẻ <br> xuống dòng inline và các trích dẫn dạng [1], [2]
  const parseInlineStyles = (text: string, sources: any[] = []) => {
    if (!text) return null;

    // Tách văn bản theo các thẻ <br> hoặc <br/>
    let parts: (string | React.ReactNode)[] = [];
    const brSplit = text.split(/<br\s*\/?>/gi);
    brSplit.forEach((subPart, i) => {
      if (i > 0) parts.push(<br key={`br-${i}`} />);
      if (subPart) parts.push(subPart);
    });

    // 1. Nhận diện **chữ đậm** (Bao gồm cả lỗi sinh markdown của LLM như *text**, **text*, *text*)
    let boldParts: (string | React.ReactNode)[] = [];
    parts.forEach((part, pIdx) => {
      if (typeof part !== "string") {
        boldParts.push(part);
        return;
      }

      // Regex mới bắt bất kỳ chuỗi nào nằm giữa 1 hoặc 2 dấu sao (asterisk)
      let boldRegex = /(?:\*\*?)([^*]+)(?:\*\*?)/g;
      let boldMatches = Array.from(part.matchAll(boldRegex));
      if (boldMatches.length > 0) {
        let lastIndex = 0;
        boldMatches.forEach((match, idx) => {
          const matchStart = match.index!;
          const matchEnd = matchStart + match[0].length;
          const textBefore = part.substring(lastIndex, matchStart);
          const boldText = match[1];

          if (textBefore) boldParts.push(textBefore);
          boldParts.push(<strong key={`b-${pIdx}-${idx}`} className="font-semibold text-slate-900">{boldText}</strong>);
          lastIndex = matchEnd;
        });
        const textAfter = part.substring(lastIndex);
        if (textAfter) boldParts.push(textAfter);
      } else {
        boldParts.push(part);
      }
    });
    parts = boldParts;

    // 2. Nhận diện các chú dẫn nguồn số như [1], [2] để biến thành các tag nhấp nháy liên kết sang sources
    let tempParts: (string | React.ReactNode)[] = [];
    parts.forEach((part, pIdx) => {
      if (typeof part !== "string") {
        tempParts.push(part);
        return;
      }

      const citeRegex = /\[(\d+)\]/g;
      const citeMatches = Array.from(part.matchAll(citeRegex));

      if (citeMatches.length > 0) {
        let lastIndex = 0;
        citeMatches.forEach((match, idx) => {
          const matchStart = match.index!;
          const matchEnd = matchStart + match[0].length;
          const textBefore = part.substring(lastIndex, matchStart);
          const citeNum = parseInt(match[1]);

          if (textBefore) tempParts.push(textBefore);

          // Tạo badge chú thích tương tác
          const sourceIdx = citeNum - 1;
          const hasSource = sources && sources[sourceIdx];

          tempParts.push(
            <span
              key={`cite-${pIdx}-${idx}`}
              onMouseEnter={() => hasSource && setHoveredCitationIdx(sourceIdx)}
              onMouseLeave={() => setHoveredCitationIdx(null)}
              onClick={() => {
                const el = document.getElementById(`source-card-${sourceIdx}`);
                el?.scrollIntoView({ behavior: "smooth", block: "center" });
              }}
              className={`inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 text-xs font-bold font-mono rounded-full cursor-pointer transition-all duration-200 ${hoveredCitationIdx === sourceIdx
                ? "bg-cyan-500 text-white shadow-sm ring-2 ring-cyan-200 scale-110"
                : "bg-blue-50 text-blue-600 hover:bg-blue-100 border border-blue-200"
                }`}
              title={hasSource ? sources[sourceIdx].ten_van_ban : `Nguồn ${citeNum}`}
            >
              {citeNum}
            </span>
          );
          lastIndex = matchEnd;
        });
        const textAfter = part.substring(lastIndex);
        if (textAfter) tempParts.push(textAfter);
      } else {
        tempParts.push(part);
      }
    });

    return <>{tempParts}</>;
  };

  // Lọc lịch sử theo ô tìm kiếm
  const filteredSessions = sessions.filter((s) =>
    s.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 font-sans">

      {/* ──────────────────────────────────────────────────────── */}
      {/* SIDEBAR BÊN TRÁI                                         */}
      {/* ──────────────────────────────────────────────────────── */}
      <aside className="w-80 h-full bg-white border-r border-slate-200 flex flex-col z-20 shadow-sm shrink-0">

        {/* Logo "AI Tra Cứu Luật" */}
        <div className="p-5 border-b border-slate-100 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-400 flex items-center justify-center shadow-md shadow-blue-200 text-white animate-pulse">
            <Scale className="w-5.5 h-5.5" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-extrabold text-[17px] tracking-tight bg-gradient-to-r from-blue-700 to-cyan-500 bg-clip-text text-transparent">
                AI TRA CỨU LUẬT
              </span>
            </div>
            <p className="text-[10px] font-medium text-slate-400 uppercase tracking-widest mt-0.5">Trợ lý Pháp lý Giáo dục</p>
          </div>
        </div>

        {/* Nút Tạo Cuộc Trò Chuyện Mới */}
        <div className="p-4">
          <button
            onClick={handleCreateNewChat}
            disabled={isStreaming}
            className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-medium text-sm flex items-center justify-center gap-2.5 shadow-md shadow-blue-100 transition-all duration-300 transform hover:-translate-y-0.5 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="w-4.5 h-4.5" />
            Cuộc trò chuyện mới
          </button>
        </div>

        {/* Ô tìm kiếm Lịch sử */}
        <div className="px-4 mb-2">
          <div className="relative">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Tìm kiếm lịch sử cuộc trò chuyện..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 text-xs rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-500 bg-slate-50 transition-all"
            />
          </div>
        </div>

        {/* Danh sách cuộc trò chuyện cũ */}
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
          <AnimatePresence initial={false}>
            {filteredSessions.length > 0 ? (
              filteredSessions.map((session) => {
                const isActive = activeSessionId === session.session_id;
                return (
                  <motion.div
                    key={session.session_id}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.2 }}
                    onClick={() => !isStreaming && setActiveSessionId(session.session_id)}
                    className={`group relative flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all duration-200 select-none ${isActive
                      ? "bg-blue-50 text-blue-800 border-l-4 border-blue-600 shadow-sm"
                      : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                      } ${isStreaming ? "opacity-60 cursor-not-allowed" : ""}`}
                  >
                    <MessageSquare className={`w-4 h-4 shrink-0 ${isActive ? "text-blue-600" : "text-slate-400"}`} />
                    <div className="flex-1 min-w-0 pr-6">
                      <p className="text-xs font-semibold truncate leading-normal">
                        {session.title || "Cuộc trò chuyện mới"}
                      </p>
                      <p className="text-[10px] text-slate-400 mt-1 font-medium">
                        {new Date(session.updated_at * 1000).toLocaleDateString("vi-VN", {
                          hour: "2-digit",
                          minute: "2-digit"
                        })}
                      </p>
                    </div>

                    {/* Nút xóa session */}
                    <button
                      onClick={(e) => handleDeleteSession(e, session.session_id)}
                      disabled={isStreaming}
                      className="absolute right-2 opacity-0 group-hover:opacity-100 hover:bg-red-50 text-slate-400 hover:text-red-600 p-1.5 rounded-lg transition-all duration-200 cursor-pointer disabled:opacity-0"
                      title="Xóa đoạn chat này"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </motion.div>
                );
              })
            ) : (
              <div className="text-center py-8 text-slate-400 text-xs font-medium">
                {searchQuery ? "Không tìm thấy kết quả" : "Chưa có cuộc hội thoại nào"}
              </div>
            )}
          </AnimatePresence>
        </div>

        {/* Chân trang Sidebar */}
        <div className="p-4 border-t border-slate-100 bg-slate-50/50 space-y-2">
          <button
            onClick={() => setShowSupportModal(true)}
            className="w-full py-2.5 px-3 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-sm cursor-pointer"
          >
            <Headphones className="w-4 h-4 text-blue-500" />
            Liên hệ hỗ trợ
          </button>
          <a
            href="https://vbpl.vn/"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full py-2.5 px-3 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-sm cursor-pointer no-underline"
          >
            <ExternalLink className="w-4 h-4 text-cyan-500" />
            Cơ sở dữ liệu VBPL
          </a>
        </div>
      </aside>

      {/* ──────────────────────────────────────────────────────── */}
      {/* KHUNG CHAT CHÍNH (Ở GIỮA VÀ BÊN PHẢI)                     */}
      {/* ──────────────────────────────────────────────────────── */}
      <main className="flex-1 h-full flex flex-col overflow-hidden relative">

        {/* Header trên cùng */}
        <header className="h-16 border-b border-slate-200 bg-white flex items-center justify-between px-6 shrink-0 z-10 shadow-sm">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping"></span>
            <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Trạng thái hệ thống: Sẵn sàng</span>
          </div>

          <div className="flex items-center gap-4">
            <div className="py-1 px-3 rounded-full bg-blue-50 border border-blue-100 text-blue-700 font-semibold text-xs flex items-center gap-1.5 cursor-default select-none shadow-sm">
              <Sparkles className="w-3.5 h-3.5 animate-spin" style={{ animationDuration: '6s' }} />
              Bản Pro ✨
            </div>

            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-slate-200 to-slate-300 flex items-center justify-center text-slate-600 border border-slate-200 text-xs font-bold shadow-sm cursor-pointer select-none hover:ring-2 hover:ring-blue-100 transition-all">
              <User className="w-4 h-4" />
            </div>
          </div>
        </header>

        {/* Khung chứa các tin nhắn */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {messages.length === 0 ? (

            /* ────────────────────────────────────────────────── */
            /* WELCOME SCREEN (TRẠNG THÁI TRỐNG)                  */
            /* ────────────────────────────────────────────────── */
            <div className="max-w-3xl mx-auto h-full flex flex-col justify-center py-10">
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-center mb-8"
              >
                {/* Logo trung tâm lớn */}
                <div className="inline-flex w-24 h-24 rounded-3xl bg-gradient-to-tr from-blue-600 via-blue-500 to-cyan-400 items-center justify-center shadow-xl shadow-blue-200 text-white mb-6 transform hover:scale-105 transition-transform duration-300">
                  <Scale className="w-12 h-12" />
                </div>
                <h2 className="text-3xl font-extrabold text-slate-800 tracking-tight leading-tight">
                  AI Tra cứu Luật có thể hỗ trợ gì cho bạn?
                </h2>
                <p className="text-slate-400 mt-2.5 text-sm font-medium">
                  Hệ thống phân tích luật Giáo dục chuyên nghiệp của Việt Nam, cung cấp nguồn chính xác và đáng tin cậy.
                </p>
              </motion.div>

              {/* Grid các câu hỏi gợi ý */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-4"
              >
                {SUGGESTIONS.map((s, idx) => (
                  <div
                    key={idx}
                    onClick={() => handleSendMessage(s.text)}
                    className="p-4 rounded-2xl bg-white border border-slate-200 hover:border-blue-400 hover:bg-blue-50/20 shadow-sm hover:shadow-md cursor-pointer transition-all duration-300 group flex flex-col justify-between"
                  >
                    <div>
                      <p className="text-[13.5px] font-bold text-slate-700 group-hover:text-blue-900 transition-colors leading-relaxed">
                        {s.text}
                      </p>
                      <p className="text-[11px] text-slate-400 mt-2 font-medium leading-relaxed">
                        {s.desc}
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-blue-600 mt-4 opacity-0 group-hover:opacity-100 transition-all duration-300 self-start">
                      Tra cứu nhanh
                      <ArrowRight className="w-3.5 h-3.5" />
                    </div>
                  </div>
                ))}
              </motion.div>
            </div>
          ) : (

            /* ────────────────────────────────────────────────── */
            /* MÀN HÌNH CHAT (DANH SÁCH TIN NHẮN CHAT)            */
            /* ────────────────────────────────────────────────── */
            <div className="max-w-4xl mx-auto space-y-6">
              {messages.map((message, index) => {
                const isUser = message.role === "user";
                return (
                  <div
                    key={index}
                    className={`flex gap-4 ${isUser ? "justify-end" : "justify-start"}`}
                  >
                    {/* Icon đại diện AI */}
                    {!isUser && (
                      <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-400 flex items-center justify-center text-white shadow-md shadow-blue-100 shrink-0 font-bold select-none self-start">
                        <Scale className="w-4.5 h-4.5" />
                      </div>
                    )}

                    {/* Bong bóng tin nhắn */}
                    <div className="max-w-[85%] flex flex-col">
                      <div
                        className={`p-5 rounded-2xl shadow-sm ${isUser
                          ? "bg-gradient-to-tr from-blue-600 to-blue-700 text-white rounded-tr-none border-b border-blue-700"
                          : "bg-white border border-slate-200 text-slate-800 rounded-tl-none"
                          }`}
                      >
                        {isUser ? (
                          <p className="text-[16.5px] leading-relaxed font-medium text-justify">{message.content}</p>
                        ) : (
                          // Trình render Markdown cho Bot
                          <div>
                            {message.content ? (
                              renderMessageContent(message.content, message.sources)
                            ) : (
                              // Loading / Pulsing state
                              <div className="flex items-center gap-2 py-2">
                                <Loader2 className="w-5.5 h-5.5 text-blue-500 animate-spin" />
                                <span className="text-slate-400 text-[15.5px] font-medium animate-pulse">
                                  {pipelineStatus || "Đang phân tích luật pháp..."}
                                </span>
                              </div>
                            )}

                            {/* Cảnh báo luật đã sửa đổi nếu có */}
                            {message.warning && (
                              <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-xl text-amber-800 text-[12px] flex gap-2.5 font-medium leading-relaxed">
                                <Info className="w-4.5 h-4.5 text-amber-600 shrink-0" />
                                <span>{message.warning}</span>
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Chi tiết kĩ thuật nhỏ ở dưới câu trả lời AI (Premium) */}
                      {!isUser && message.content && (
                        <div className="flex flex-wrap items-center gap-3.5 mt-2 px-1 text-[11px] text-slate-400 font-semibold">
                          {message.is_pro ? (
                            <span className="flex items-center gap-1 bg-gradient-to-r from-violet-500/10 to-indigo-500/10 border border-indigo-200/50 py-0.5 px-2.5 rounded-full text-indigo-600 font-bold scale-95 uppercase select-none">
                              <Sparkles className="w-3 h-3 text-indigo-500 animate-pulse" />
                              GPT 120B (Pro) ⚡
                            </span>
                          ) : (
                            <span className="flex items-center gap-1 bg-slate-100 py-0.5 px-2.5 rounded-full font-mono text-slate-500 scale-95 uppercase select-none">
                              GPT 120B (Standard)
                            </span>
                          )}
                          {message.intent && (
                            <span className="flex items-center gap-1 bg-slate-100 py-0.5 px-2 rounded-full font-mono text-slate-500 scale-95 uppercase">
                              <Compass className="w-3 h-3 text-slate-400" />
                              Ý định: {message.intent}
                            </span>
                          )}
                          {message.latency_ms !== undefined && (
                            <span className="flex items-center gap-1 bg-slate-100 py-0.5 px-2 rounded-full font-mono text-slate-500 scale-95">
                              <Clock className="w-3 h-3 text-slate-400" />
                              Độ trễ: {(message.latency_ms / 1000).toFixed(2)}s
                            </span>
                          )}
                        </div>
                      )}

                      {/* ──────────────────────────────────────────────── */}
                      {/* VĂN BẢN TRÍCH DẪN (SOURCES)                      */}
                      {/* ──────────────────────────────────────────────── */}
                      {!isUser && message.sources && message.sources.length > 0 && (
                        <div className="mt-4 space-y-2.5">
                          <p className="text-[14px] font-bold text-blue-900 uppercase tracking-widest pl-1">
                            Văn bản luật trích dẫn ({message.sources.length}):
                          </p>
                          <div className="grid grid-cols-1 gap-2.5">
                            {message.sources.map((src, srcIdx) => {
                              const isHighlighted = hoveredCitationIdx === srcIdx;
                              return (
                                <div
                                  id={`source-card-${srcIdx}`}
                                  key={srcIdx}
                                  className={`p-4 rounded-xl border transition-all duration-300 ${isHighlighted
                                    ? "bg-cyan-50 border-cyan-400 shadow-md ring-2 ring-cyan-100 transform -translate-y-0.5"
                                    : "bg-white border-slate-200 hover:border-slate-300 shadow-sm"
                                    }`}
                                >
                                  <div className="flex items-start justify-between gap-3">
                                    <div className="flex items-center gap-2">
                                      <span className={`w-6.5 h-6.5 rounded-full flex items-center justify-center text-[12px] font-bold font-mono ${isHighlighted ? "bg-cyan-500 text-white" : "bg-blue-50 text-blue-600"
                                        }`}>
                                        {srcIdx + 1}
                                      </span>
                                      <p className="text-[14.5px] font-bold text-slate-800 leading-tight">
                                        {src.ten_van_ban || "Văn bản chưa xác định"}
                                      </p>
                                    </div>
                                    <span className="text-[11px] font-bold px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 uppercase tracking-wider scale-90">
                                      {src.loai_van_ban || "Văn bản"}
                                    </span>
                                  </div>

                                  <div className="mt-2.5 pl-7">
                                    {src.so_dieu && (
                                      <div className="text-[13px] font-bold text-blue-700 bg-blue-50/50 inline-block px-2.5 py-0.5 rounded border border-blue-100">
                                        Điều {src.so_dieu}
                                      </div>
                                    )}
                                    <p className="text-[14.5px] text-slate-600 leading-relaxed mt-2 text-justify italic bg-slate-50/50 p-2.5 rounded-lg border border-dashed border-slate-200">
                                      "{src.content || src.text || "Nội dung quy định"}"
                                    </p>
                                  </div>

                                  {src.tinh_trang && (() => {
                                    const tinhTrangLower = src.tinh_trang.toLowerCase();
                                    let colorClass = "bg-slate-400";
                                    let labelText = src.tinh_trang;

                                    if (tinhTrangLower === "con_hieu_luc" || tinhTrangLower === "con-hieu-luc" || tinhTrangLower === "con hieu luc") {
                                      colorClass = "bg-emerald-500";
                                      labelText = "Còn hiệu lực";
                                    } else if (tinhTrangLower === "da_sua_doi" || tinhTrangLower === "da-sua-doi" || tinhTrangLower === "da sua doi") {
                                      colorClass = "bg-amber-500";
                                      labelText = "Đã sửa đổi";
                                    } else if (tinhTrangLower === "bi_bai_bo" || tinhTrangLower === "bi-bai-bo" || tinhTrangLower === "bi bai bo") {
                                      colorClass = "bg-rose-500";
                                      labelText = "Bị bãi bỏ";
                                    } else if (tinhTrangLower === "het_hieu_luc" || tinhTrangLower === "het-hieu-luc" || tinhTrangLower === "het hieu luc") {
                                      colorClass = "bg-rose-500";
                                      labelText = "Hết hiệu lực";
                                    } else if (tinhTrangLower === "sap_co_hieu_luc" || tinhTrangLower === "sap-co-hieu-luc" || tinhTrangLower === "sap co hieu luc") {
                                      colorClass = "bg-blue-500";
                                      labelText = "Sắp có hiệu lực";
                                    } else if (tinhTrangLower === "chua_co_hieu_luc" || tinhTrangLower === "chua-co-hieu-luc" || tinhTrangLower === "chua co hieu luc") {
                                      colorClass = "bg-slate-400";
                                      labelText = "Chưa có hiệu lực";
                                    } else if (tinhTrangLower === "het_hieu_luc_mot_phan" || tinhTrangLower === "het-hieu-luc-mot-phan" || tinhTrangLower === "het hieu luc mot phan") {
                                      colorClass = "bg-orange-500";
                                      labelText = "Hết hiệu lực một phần";
                                    } else {
                                      labelText = src.tinh_trang.replace(/_/g, ' ');
                                      labelText = labelText.charAt(0).toUpperCase() + labelText.slice(1);
                                    }

                                    return (
                                      <div className="mt-2.5 pl-7 flex items-center gap-1.5">
                                        <span className={`w-1.5 h-1.5 rounded-full ${colorClass}`}></span>
                                        <span className="text-[11.5px] font-semibold text-slate-400">
                                          Hiệu lực: {labelText}
                                        </span>
                                      </div>
                                    );
                                  })()}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Icon người dùng */}
                    {isUser && (
                      <div className="w-9 h-9 rounded-xl bg-blue-100 border border-blue-200 flex items-center justify-center text-blue-600 font-bold shrink-0 shadow-sm select-none self-start">
                        <User className="w-4.5 h-4.5" />
                      </div>
                    )}
                  </div>
                );
              })}

              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {/* ──────────────────────────────────────────────────────── */}
        {/* THANH NHẬP CHAT (ĐẶT Ở DƯỚI CÙNG)                         */}
        {/* ──────────────────────────────────────────────────────── */}
        <footer className="p-5 bg-gradient-to-t from-slate-50 via-slate-50/95 to-transparent shrink-0">
          <div className="max-w-3xl mx-auto">

            {/* Input container */}
            <div className="bg-white border border-slate-200 rounded-2xl shadow-lg focus-within:ring-4 focus-within:ring-blue-100 focus-within:border-blue-500 transition-all duration-300 overflow-hidden">

              {/* Textarea nhập liệu */}
              <textarea
                ref={inputRef}
                rows={1}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value.substring(0, 2000))}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder="Nhập câu hỏi pháp luật của bạn tại đây... (Ví dụ: lộ trình nâng chuẩn giáo viên)"
                className="w-full px-5 pt-4 pb-2 text-sm bg-transparent border-0 focus:outline-none resize-none min-h-[50px] max-h-[160px] text-slate-800 leading-relaxed placeholder-slate-400"
                style={{ height: "auto" }}
              />

              {/* Bottom bar inside input container */}
              <div className="px-5 py-3 border-t border-slate-100 bg-slate-50/50 flex items-center justify-between">

                {/* Nút cài đặt nhanh AI Pro */}
                <div className="flex items-center gap-2">
                  <div
                    onClick={() => setIsProMode(!isProMode)}
                    title={isProMode ? "Đang sử dụng mô hình GPT 120B (Pro)" : "Đang sử dụng mô hình GPT 120B (Thường)"}
                    className={`py-1 px-2.5 rounded-full font-bold text-[10px] tracking-wide uppercase flex items-center gap-1.5 select-none shadow-sm cursor-pointer transition-all duration-300 ${
                      isProMode
                        ? "bg-gradient-to-r from-violet-600 to-indigo-600 text-white border-violet-500 scale-105 shadow-md shadow-indigo-100 hover:opacity-90 animate-pulse border"
                        : "bg-blue-50 border border-blue-100 text-blue-600 hover:bg-blue-100"
                    }`}
                  >
                    <Sparkles 
                      className={`w-3 h-3 ${isProMode ? "text-amber-300 animate-spin" : "text-blue-500"}`} 
                      style={isProMode ? { animationDuration: '4s' } : undefined}
                    />
                    AI Pro {isProMode ? "ON ✨" : "OFF 💤"}
                  </div>
                  {activeIntent && (
                    <div className="py-1 px-2.5 rounded-full bg-slate-100 border border-slate-200 text-slate-500 font-mono text-[9px] uppercase select-none scale-95">
                      Ý định: {activeIntent}
                    </div>
                  )}
                </div>

                {/* Bộ đếm ký tự & Nút Gửi */}
                <div className="flex items-center gap-3">
                  <span className="text-[11px] font-semibold font-mono text-slate-400 select-none">
                    {inputValue.length}/2000
                  </span>

                  <button
                    onClick={() => handleSendMessage()}
                    disabled={!inputValue.trim() || isStreaming}
                    className={`w-9 h-9 rounded-xl flex items-center justify-center shadow-md transition-all duration-300 cursor-pointer ${inputValue.trim() && !isStreaming
                      ? "bg-gradient-to-tr from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white shadow-blue-100 hover:shadow-lg transform hover:-translate-y-0.5"
                      : "bg-slate-100 text-slate-400 cursor-not-allowed shadow-none"
                      }`}
                  >
                    {isStreaming ? (
                      <Loader2 className="w-4.5 h-4.5 animate-spin" />
                    ) : (
                      <Send className="w-4.5 h-4.5" />
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Điều khoản sử dụng nhỏ ở dưới */}
            <p className="text-[10.5px] text-slate-400 text-center font-medium mt-3 leading-relaxed">
              Thông tin được tạo ra bởi trí tuệ nhân tạo. Hãy luôn cẩn trọng và kiểm chứng nguồn gốc. Xem thêm về{" "}
              <a href="#" className="text-blue-500 hover:underline">Quyền riêng tư</a> và{" "}
              <a href="#" className="text-blue-500 hover:underline">Điều khoản LawEdu AI</a>.
            </p>
          </div>
        </footer>
      </main>

      {/* Hộp thoại thông tin liên hệ hỗ trợ dạng Modal nổi cực đẹp */}
      <AnimatePresence>
        {showSupportModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowSupportModal(false)}
            className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center z-50 p-4"
          >
            <motion.div
              initial={{ scale: 0.95, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 20 }}
              transition={{ type: "spring", duration: 0.4 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white rounded-2xl p-6 max-w-sm w-full shadow-2xl border border-slate-100 text-center relative overflow-hidden"
            >
              {/* Icon tai nghe phát sáng */}
              <div className="mx-auto w-14 h-14 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center mb-4 shadow-sm border border-blue-100 animate-pulse">
                <Headphones className="w-6 h-6 animate-bounce" />
              </div>

              <h3 className="text-lg font-bold text-slate-800 tracking-tight">Liên hệ hỗ trợ</h3>
              <p className="text-slate-500 mt-2 text-[13.5px] leading-relaxed font-medium">
                Mọi thắc mắc, phản hồi hoặc cần hỗ trợ về ứng dụng, xin vui lòng liên hệ với tôi qua địa chỉ email:
              </p>

              {/* Thẻ hiển thị Email có thể click để bôi đen/copy nhanh */}
              <div className="mt-4 p-3 bg-slate-50 border border-slate-200 rounded-xl flex items-center justify-center gap-2 select-all hover:bg-slate-100 transition-colors cursor-pointer" title="Bôi đen và Copy">
                <span className="font-bold text-[14.5px] text-blue-900 font-mono">
                  buihaidang112299@gmail.com
                </span>
              </div>

              {/* Nút đóng */}
              <button
                onClick={() => setShowSupportModal(false)}
                className="w-full mt-5 py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-900 text-white font-semibold text-xs transition-all shadow-md cursor-pointer"
              >
                Đóng
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
