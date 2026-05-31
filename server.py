from fastmcp import FastMCP
from backend.app import appp

mcp = FastMCP.from_fastapi(
    app=appp,
    name="Research-paper-MCP-server"
)

if __name__=="__main__":
    mcp.run()