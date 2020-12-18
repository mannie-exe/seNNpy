
const path = require('path')
const fs = require('fs')
const fsPromises = fs.promises


const cleanAndSaveMessages = async messages => {
  try {
    const data = JSON.parse(await fsPromises.readFile(messages))

    const cleanMessages = data.messages.reduce((accum, message) => {
      // remove links
      const cleanMessage = message.content.replace(
        /(?:https?|ftp):\/\/[\n\S]+/g,
        ''
      )
      // skip empty
      if (cleanMessage) accum.push(cleanMessage)
      return accum
    }, [])

    await fsPromises.writeFile(path.resolve('./messages.txt'), cleanMessages.join('\n'))
  } catch (e) {
    console.error(e)
    throw e.code
  }
};

cleanAndSaveMessages(path.resolve('./export.json'))
