import React, { useState } from 'react';
import axios from 'axios';
import { CreditCard, Banknote, BarChart, TrendingUp } from 'lucide-react';

export default function CreditScoringApp() {
  const [formData, setFormData] = useState({
    Name: 'Test User', ssn: '123-45-6789', Age: '30', Occupation: 'Engineer', Annual_Income: '16000',
    Monthly_Inhand_Salary: '1787', Num_Bank_Accounts: '4', Num_Credit_Card: '5',
    Interest_Rate: '25', Num_of_Loan: '2', Type_of_Loan: 'Home Loan',
    Delay_from_due_date: '2', Num_of_Delayed_Payment: '7', Credit_Mix: 'Fair',
    Outstanding_Debt: '7175', Credit_Utilization_Ratio: '25',
    Credit_History_Age: '120', Total_EMI_per_month: '300'
  });
  const [creditScore, setCreditScore] = useState(null);
  const [activeTab, setActiveTab] = useState('status');
  const [scoreBreakdown, setScoreBreakdown] = useState({ repayment: 0, utilization: 0, outstanding: 0, inquiries: 0 });
  const [recommendations, setRecommendations] = useState([]);
  const [summary, setSummary] = useState('');
  const [vectorProducts, setVectorProducts] = useState([]);
  const [anomalyScore, setAnomalyScore] = useState(null);
  const ANOMALY_THRESHOLD = 0.7;

  const handleChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    setVectorProducts([]);
    e.preventDefault();
    const emptyField = Object.entries(formData).find(([_, v]) => v.trim() === "");
    if (emptyField) {
      alert(`Please fill in all fields. Missing: ${emptyField[0]}`);
      return;
    }

    try {
      const res = await axios.post('http://localhost:8000/score', formData);
      const result = res.data;
      if (result.status === 'rejected' || result.status === 'flagged') {
        const reason = result.description || result.reason || (result.flags ? result.flags.join(', ') : 'No details provided');
        alert(`${result.status === 'rejected' ? 'Application rejected' : 'Application flagged'}: ${reason}`);
        return;
      }

      setCreditScore(result.credit_score_estimate);
      setAnomalyScore(result.anomaly_score ?? result.fraud_risk ?? null);
      setScoreBreakdown({
        repayment: result.repayment || 0,
        utilization: result.utilization || 0,
        outstanding: result.outstanding || 0,
        inquiries: result.inquiries || 0
      });
      setRecommendations(result.recommendations || []);
      setSummary(result.summary || '');
      // Optional: fetch vector recommendations
      try {
        const vecRes = await axios.post('http://localhost:8000/similar_products', {
          description: `Customer profile with income ${formData.Annual_Income}, occupation ${formData.Occupation}, utilization ${formData.Credit_Utilization_Ratio}, and credit mix ${formData.Credit_Mix}`
        });
        setVectorProducts(vecRes.data.results || []);
      } catch (e) {
        console.warn('Vector search failed:', e.message);
      }
    } catch (error) {
      console.error('Error posting to backend:', error.response?.data || error.message);
      alert("Error submitting form. Check the console for details.");
    }
  };

  const sliderFields = ['Outstanding_Debt', 'Num_Credit_Card', 'Num_Bank_Accounts', 'Total_EMI_per_month', 'Monthly_Inhand_Salary', 'Num_of_Delayed_Payment'];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 min-h-screen bg-gradient-to-br from-green-50 via-white to-green-100">
      {/* Left Panel with Image */}
      <div className="bg-gradient-to-br from-green-600 to-green-400 flex items-center justify-center p-6">
        <img src="/credit-card-image.jpg" alt="Credit Visual" className="max-w-full max-h-full rounded-xl shadow-xl" />
      </div>

      {/* Right Panel with Form and Dashboard */}
      <div className="p-6 lg:p-12 overflow-y-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">AI Credit Scoring Dashboard</h1>

        <form className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10" onSubmit={handleSubmit}>
          {Object.entries(formData).map(([key, value]) => (
            <div key={key}>
              <label className="block text-sm font-semibold mb-1 text-gray-700">{key === 'ssn' ? 'SSN' : key.replaceAll('_', ' ')}</label>
              {sliderFields.includes(key) ? (
                <input
                  type="range"
                  name={key}
                  value={value}
                  onChange={handleChange}
                  className="w-full"
                  min="0"
                  max="10000"
                />
              ) : (
                <input
                  type="text"
                  name={key}
                  value={value}
                  onChange={handleChange}
                  className="w-full border px-3 py-2 rounded shadow-sm"
                />
              )}
              {sliderFields.includes(key) && <span className="text-sm text-gray-600">{value}</span>}
            </div>
          ))}
          <div className="md:col-span-2 flex justify-center">
            <button
              type="submit"
              className="bg-green-600 text-white px-8 py-3 rounded shadow-md hover:bg-green-700 transition-colors duration-200"
            >
              Save Profile
            </button>
          </div>
        </form>

        {creditScore && creditScore < 600 ? (
          <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">
            Credit Health Status : <span className="bg-red-100 text-red-700 px-4 py-1 rounded-full ml-2">REJECTED</span>
          </h2>
        ) : (
          <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">
            Credit Health Status : <span className="bg-green-100 text-green-700 px-4 py-1 rounded-full ml-2">APPROVED</span>
          </h2>
        )}

        {anomalyScore !== null && (
          <>
            <h2 className="text-2xl font-bold text-center text-gray-800 mb-2">
              Fraud Risk: <span className={`px-4 py-1 rounded-full ml-2 ${anomalyScore > ANOMALY_THRESHOLD ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>{anomalyScore.toFixed(2)}</span>
            </h2>
            {anomalyScore > ANOMALY_THRESHOLD && (
              <p className="text-center text-red-700 mb-4">⚠️ High anomaly score detected</p>
            )}
          </>
        )}

        <div className="flex justify-center space-x-6 mb-8">
          <button
            className={`px-4 py-2 rounded shadow transition-colors duration-200 ${activeTab === 'status' ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
            onClick={() => setActiveTab('status')}
          >
            Status explanation
          </button>
          <button
            className={`px-4 py-2 rounded shadow transition-colors duration-200 ${activeTab === 'products' ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
            onClick={() => setActiveTab('products')}
          >
            Product recommendations
          </button>
        </div>

        {activeTab === 'status' && (
          <div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center text-sm font-medium text-gray-700 mb-6">
              <div className="bg-white rounded shadow p-4">
                <BarChart className="mx-auto mb-1 text-green-600" />
                <div className="text-2xl font-bold text-green-700">{scoreBreakdown.repayment}</div>
                <div>Repayment History</div>
              </div>
              <div className="bg-white rounded shadow p-4">
                <TrendingUp className="mx-auto mb-1 text-green-600" />
                <div className="text-2xl font-bold text-green-700">{scoreBreakdown.utilization}</div>
                <div>Credit Utilization</div>
              </div>
              <div className="bg-white rounded shadow p-4">
                <Banknote className="mx-auto mb-1 text-green-600" />
                <div className="text-2xl font-bold text-green-700">{scoreBreakdown.outstanding}</div>
                <div>Outstanding</div>
              </div>
              <div className="bg-white rounded shadow p-4">
                <CreditCard className="mx-auto mb-1 text-green-600" />
                <div className="text-2xl font-bold text-green-700">{scoreBreakdown.inquiries}</div>
                <div>Num Credit Inquiries</div>
              </div>
            </div>

            {summary && (
              <div className="bg-white rounded shadow p-6 text-sm text-gray-700 max-w-4xl mx-auto">
                <h2 className="text-md font-semibold mb-2 text-gray-800">LLM Credit Risk Summary</h2>
                <p dangerouslySetInnerHTML={{ __html: summary.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}></p>
              </div>
            )}
          </div>
        )}

            {activeTab === 'products' && (
          <>
            <div className="bg-white rounded shadow p-6 text-sm text-gray-700 max-w-4xl mx-auto mt-6">
              
            <p>Based on your credit profile, we recommend exploring:</p>
            <ul className="list-disc list-inside mt-2">
              {recommendations.length > 0
                ? recommendations.map((rec, idx) => <li key={idx}>{rec}</li>)
                : <li>No personalized recommendations yet.</li>
              }
            </ul>
              
              <p>🔍 AI-Suggested Credit Card Products for You:</p>
              <ul className="divide-y divide-gray-200 mt-3">
{vectorProducts.length > 0 ? (
                vectorProducts.map((prod, idx) => (
                  <li className="py-2" key={idx}>
                    <h4 className="text-green-700 font-semibold">{prod.title}</h4>
                    <p className="text-xs">{prod.text}</p>
                    <a href={prod.source} target="_blank" className="text-xs text-blue-600 underline">View Details</a>
                  </li>
                ))
              ) : (
                <li className="text-xs text-gray-500">No AI product suggestions yet.</li>
              )}
              </ul>
            </div>
          </>
        )}
          </div>
        </div>
  );
}
