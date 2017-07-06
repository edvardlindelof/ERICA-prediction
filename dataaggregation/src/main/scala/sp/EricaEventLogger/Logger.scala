package sp.EricaEventLogger

import akka.persistence._
import sp.gPubSub.API_Data.EricaEvent

class Logger(recoveredEventHandler: RecoveredEventHandler = PrintingHandler) extends PersistentActor {
  override def persistenceId = "EricaEventLogger"

  override def receiveCommand = {
    case ev: EricaEvent => persist(ev)(ev => println("EricaEventLogger persisted " + ev))
  }

  override def receiveRecover = {
    case ev: EricaEvent => recoveredEventHandler.handleEvent(ev)
    case RecoveryCompleted => context.system.terminate()
  }
}

trait RecoveredEventHandler {
  def handleEvent(ev: EricaEvent): Unit
}

object PrintingHandler extends RecoveredEventHandler {
  override def handleEvent(ev: EricaEvent) = println("EricaEventLogger recovered " + ev)
}
