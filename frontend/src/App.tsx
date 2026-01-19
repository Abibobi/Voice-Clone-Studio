import React, { useState, useRef } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const handleSubmit = async () => {
    if (!text) return;
    setLoading(true);
    setAudioUrl(null);
    setStatus('Queued...');

    try {
      const res = await axios.post(`${API_BASE}/tts`, { text });
      const jobId = res.data.job_id;
      setStatus(`Job ID: ${jobId} - Processing...`);

      pollRef.current = setInterval(async () => {
        try {
          const pollRes = await axios.get(`${API_BASE}/job/${jobId}`);
          const data = pollRes.data;

          if (data.status === 'finished') {
            if (pollRef.current) clearInterval(pollRef.current);
            setAudioUrl(`${API_BASE}/static/${data.result}`);
            setStatus('Ready');
            setLoading(false);
          } else if (data.status === 'failed') {
            if (pollRef.current) clearInterval(pollRef.current);
            setStatus(`Failed: ${data.error}`);
            setLoading(false);
          } else {
            setStatus(`Status: ${data.status}`);
          }
        } catch (pollErr) {
          console.error("Polling error", pollErr);
        }
      }, 1000);

    } catch (error) {
      console.error(error);
      setLoading(false);
      setStatus('Error connecting to backend');
    }
  };

  // Allow Ctrl+Enter to submit
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.ctrlKey && e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div style={{ padding: '2rem', maxWidth: '600px', margin: '0 auto', fontFamily: 'sans-serif' }}>
      <h1 style={{ color: '#333' }}>DocuVoice MVP</h1>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <textarea 
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter text to narrate here... (Ctrl + Enter to submit)"
          style={{ padding: '12px', fontSize: '16px', width: '100%', borderRadius: '4px', border: '1px solid #ccc' }}
        />
        
        <button 
          onClick={handleSubmit} 
          disabled={loading}
          style={{ 
            padding: '12px', 
            fontSize: '16px', 
            cursor: loading ? 'not-allowed' : 'pointer',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px'
          }}
        >
          {loading ? (
            <>
              <div className="spinner"></div>
              <span>Processing...</span>
            </>
          ) : (
            'Generate Voice'
          )}
        </button>
      </div>

      <div style={{ marginTop: '2rem' }}>
        <div style={{color: '#666', fontSize: '0.9rem'}}>System Status: {status}</div>
        
        {audioUrl && (
          <div style={{ marginTop: '1.5rem', padding: '1.5rem', background: '#f8f9fa', borderRadius: '8px', border: '1px solid #e9ecef' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <p style={{margin: 0, fontWeight: 'bold', color: 'green'}}>✅ Audio Generated</p>
              
              {/* DOWNLOAD BUTTON */}
              <a 
                href={audioUrl} 
                download={`docuvoice_${Date.now()}.wav`}
                style={{
                  textDecoration: 'none',
                  color: '#007bff',
                  fontWeight: 'bold',
                  fontSize: '0.9rem',
                  border: '1px solid #007bff',
                  padding: '5px 10px',
                  borderRadius: '4px'
                }}
              >
                ⬇ Download WAV
              </a>
            </div>

            <audio controls autoPlay src={audioUrl} style={{ width: '100%' }}>
              Your browser does not support the audio element.
            </audio>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;