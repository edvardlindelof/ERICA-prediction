package sp.EricaEventLogger

import akka.persistence._
import akka.actor.ActorRef
import sp.gPubSub.API_Data.EricaEvent

class Logger(recoveredEventHandler: ActorRef = null) extends PersistentActor {
  override def persistenceId = "EricaEventLogger"

  override def receiveCommand = {
    case ev: EricaEvent => persist(ev)(ev => println("EricaEventLogger persisted " + ev))
  }

  override def receiveRecover = {
    case ev: EricaEvent =>
      if(recoveredEventHandler != null) recoveredEventHandler ! ev
      else println("EricaEventLogger recovered " + ev)
    case RecoveryCompleted => context.system.terminate()
  }
}
