function read_bin_file(path)
    local file, errorMessage = io.open(path, "rb")
    if not file then 
        error("Could not read the file:" .. errorMessage .. "\n")
    end

    local content = file:read "*all"
    file:close()
    return content
end

local Boundary = "----WebKitFormBoundaryePkpFF7tjBAqx29L"
local BodyBoundary = "--" .. Boundary
local LastBoundary = "--" .. Boundary .. "--"

local CRLF = "\r\n"

local Filename = "0166.png"
local FileBody = read_bin_file(Filename)
FileBody = 0

local ContentDisposition = "Content-Disposition: form-data; name=\"content\"; filename=\"" .. Filename .. "\""
print(ContentDisposition)
local ContentDisposition = "Content-Disposition: form-data; name=\"imageName\""
print(ContentDisposition)


-- wrk.method = "POST"
-- wrk.headers["Content-Type"] = "multipart/form-data; boundary=" .. Boundary
-- wrk.body = BodyBoundary .. CRLF .. ContentDisposition .. CRLF .. CRLF .. FileBody .. CRLF .. LastBoundary

-- local headers = "multipart/form-data; boundary=" .. Boundary
-- print(headers)
local body = BodyBoundary .. CRLF .. ContentDisposition .. CRLF .. CRLF .. FileBody .. CRLF .. LastBoundary
print(body)