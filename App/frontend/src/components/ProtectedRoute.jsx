import { Navigate } from "react-router-dom";

const ProtectedRoute = ({ children }) => {
  const agentId = localStorage.getItem("agent_id");

  if (!agentId) {
    return <Navigate to="/login" replace />;
  }

  return children;
};

export default ProtectedRoute;
