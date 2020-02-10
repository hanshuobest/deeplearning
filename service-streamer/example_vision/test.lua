local function writeFile(fileName,content)
    local f = assert(io.open(fileName,'w'))
    f:write(content)
    f:close()
end

writeFile('test.txt','5.Java')
