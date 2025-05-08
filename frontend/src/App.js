import React, { useState } from 'react';
import axios from 'axios';
import { CreditCard, Banknote, BarChart, TrendingUp, User } from 'lucide-react';

export default function CreditScoringApp() {
  const [formData, setFormData] = useState({
    Name: '',
    Age: '',
    Occupation: '',
    Annual_Income: '',
    Monthly_Inhand_Salary: '',
    Num_Bank_Accounts: '',
    Num_Credit_Card: '',
    Interest_Rate: '',
    Num_of_Loan: '',
    Type_of_Loan: '',
    Delay_from_due_date: '',
    Num_of_Delayed_Payment: '',
    Credit_Mix: '',
    Outstanding_Debt: '',
    Credit_Utilization_Ratio: '',
    Credit_History_Age: '',
    Total_EMI_per_month: ''
  });

  const [creditScore, setCreditScore] = useState(null);
  const [activeTab, setActiveTab] = useState('status');
  const [scoreBreakdown, setScoreBreakdown] = useState({
    repayment: 0,
    utilization: 0,
    outstanding: 0,
    inquiries: 0
  });
  const [recommendations, setRecommendations] = useState([]);
  const [summary, setSummary] = useState('');

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await axios.post('http://localhost:8000/score', formData);
    const result = res.data;
    setCreditScore(result.credit_score_estimate);
    setScoreBreakdown({
      repayment: result.repayment || 0,
      utilization: result.utilization || 0,
      outstanding: result.outstanding || 0,
      inquiries: result.inquiries || 0
    });
    setRecommendations(result.recommendations || []);
    setSummary(result.summary || '');
  };

  const sliderFields = [
    'Outstanding_Debt',
    'Num_Credit_Card',
    'Num_Bank_Accounts',
    'Total_EMI_per_month',
    'Monthly_Inhand_Salary',
    'Num_of_Delayed_Payment'
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      <aside className="w-1/3 bg-white p-6 shadow-md border-r overflow-y-scroll">
        <div className="mb-6">
          <div className="text-xl font-semibold flex items-center gap-2"><User className="w-5 h-5" /> Rick Rothackerj</div>
          <div className="text-sm text-gray-600">Occupation: Teacher</div>
          <div className="text-sm text-gray-600">Age: 28 years</div>
          <div className="text-sm text-gray-600">Customer ID: 8625</div>
        </div>
        <form className="space-y-4" onSubmit={handleSubmit}>
          {Object.entries(formData).map(([key, value]) => (
            <div key={key}>
              <label className="block text-xs font-medium text-gray-600 mb-1">{key.replaceAll('_', ' ')}</label>
              {sliderFields.includes(key) ? (
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    name={key}
                    value={value}
                    onChange={handleChange}
                    className="w-full"
                    min="0"
                    max="10000"
                  />
                  <span className="text-sm text-gray-700">{value}</span>
                </div>
              ) : (
                <input
                  type="text"
                  name={key}
                  value={value}
                  onChange={handleChange}
                  className="border rounded w-full px-2 py-1 text-sm"
                />
              )}
            </div>
          ))}
          <button
            type="submit"
            className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 mt-4"
          >
            Save Profile
          </button>
        </form>
      </aside>

      <main className="flex-1 p-10">
        <h1 className="text-2xl font-bold mb-6">Credit Health Status: <span className="text-green-600">APPROVED</span></h1>

        <div className="flex border-b mb-4">
          <button
            className={`px-4 py-2 text-sm font-medium ${activeTab === 'status' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500'}`}
            onClick={() => setActiveTab('status')}
          >
            Status explanation
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium ${activeTab === 'products' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500'}`}
            onClick={() => setActiveTab('products')}
          >
            Product recommendations
          </button>
        </div>

        {activeTab === 'status' && (
          <>
            <div className="grid grid-cols-6 gap-4 text-center text-sm font-medium text-gray-700 mb-6">
              <div className="bg-white rounded shadow p-4">
                <BarChart className="mx-auto mb-1 text-green-600" />
                <div className="text-2xl font-bold text-green-700">{scoreBreakdown.repayment}</div>
                <div>Repayment History</div>
              </div>
              <div className="text-2xl text-gray-400">+</div>
              <div className="bg-white rounded shadow p-4">
                <TrendingUp className="mx-auto mb-1 text-green-600" />
                <div className="text-2xl font-bold text-green-700">{scoreBreakdown.utilization}</div>
                <div>Credit Utilization</div>
              </div>
              <div className="text-2xl text-gray-400">+</div>
              <div className="bg-white rounded shadow p-4">
                <Banknote className="mx-auto mb-1 text-green-600" />
                <div className="text-2xl font-bold text-green-700">{scoreBreakdown.outstanding}</div>
                <div>Outstanding</div>
              </div>
              <div className="text-2xl text-gray-400">+</div>
              <div className="bg-white rounded shadow p-4">
                <CreditCard className="mx-auto mb-1 text-green-600" />
                <div className="text-2xl font-bold text-green-700">{scoreBreakdown.inquiries}</div>
                <div>Num Credit Inquiries</div>
              </div>
              <div className="text-2xl text-gray-400">=</div>
              <div className="bg-white rounded shadow p-4 col-span-2">
                <div className="text-2xl font-bold text-green-700">{creditScore || '0'}</div>
                <div>Credit Score</div>
              </div>
            </div>
            {summary && (
              <div className="bg-white rounded shadow p-6 text-sm text-gray-700">
                <h2 className="text-md font-semibold mb-2 text-gray-800">LLM Credit Risk Summary</h2>
                <p>{summary}</p>
              </div>
            )}
          </>
        )}

        {activeTab === 'products' && (
          <div className="bg-white rounded shadow p-6 text-sm text-gray-700">
            <p>Based on your credit profile, we recommend exploring:</p>
            <ul className="list-disc list-inside mt-2">
              {recommendations.length > 0 ? (
                recommendations.map((rec, idx) => <li key={idx}>{rec}</li>)
              ) : (
                <li>No personalized recommendations yet.</li>
              )}
            </ul>
          </div>
        )}
      </main>
    </div>
  );
}
