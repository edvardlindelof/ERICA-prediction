package sp.EricaEventLogger

import akka.persistence._
import org.joda.time.DateTime
import com.github.tototoshi.csv.CSVWriter
import java.io.File
import sp.gPubSub.API_Data.EricaEvent

import scala.collection.mutable.ListBuffer

class Logger(stateKeeper: StateKeeper = PrintingStateKeeperDummy,
             futureTeller: FutureTeller = UselessFutureTellerDummy,
             name: String = "Logger") extends PersistentActor {

  override def persistenceId = "EricaEventLogger"

  override def receiveCommand = {
    case ev: EricaEvent => persist(ev)(ev => println("EricaEventLogger persisted " + ev))
  }

  val playback = ListBuffer[EricaEvent]()

  override def receiveRecover = {
    case ev: EricaEvent => playback.append(ev)
    case RecoveryCompleted => { handlePlayback(playback); context.system.terminate() }
  }

  implicit def stringToDateTime(s: String) = DateTime.parse(s)

  val samplingIntervalMins = 10
  val flushoutHours = 24 // no data is written before this has passed, bc rolling avgs etc are missing in the beginning
  var firsEventTime: DateTime = null
  var nextSampleTime: DateTime = null

  def handlePlayback(pb: ListBuffer[EricaEvent]): Unit = {
    var remainingEvents = pb.toList

    var done = false
    while(!done) {
      val ev = remainingEvents.head
      stateKeeper.handleEvent(ev)

      if(firsEventTime == null) firsEventTime = ev.Start

      if (nextSampleTime == null) nextSampleTime = ev.Start.plusMinutes(samplingIntervalMins)
      else if (ev.Start.isAfter(nextSampleTime)) {
        val timeTuple = ("epochseconds" -> (nextSampleTime.getMillis / 1000).toInt)
        val futureValues = futureTeller.futureState(remainingEvents)
        if(!futureValues.isEmpty) {
          val toWrite = timeTuple :: stateKeeper.state(nextSampleTime) ::: futureValues
          if(nextSampleTime.isAfter(firsEventTime.plusHours(flushoutHours))) {
            println(toWrite)
            WriteToCSV(name, toWrite)
          }
          nextSampleTime = nextSampleTime.plusMinutes(samplingIntervalMins)
        } else {
          done = true
        }
      }

      remainingEvents = remainingEvents.tail

      if(remainingEvents.isEmpty) done = true
    }
  }
}

// a StateKeeper takes events one by one and keeps track of the state-stuff based on information up until the latest one
// an instance of this will be copy-pasted into core erica code for live prediction
// it is convenient for generating training data because it prevents us from accidentally feeding future information
// into the features
trait StateKeeper {
  def handleEvent(ev: EricaEvent): Unit
  def state(sampleTime: DateTime): List[(String, Int)] // e.g. "ttt30" -> 1200, List just to be certain of the order
}

// takes future events and generates an output point to add to training data
trait FutureTeller {
  def futureState(events: List[EricaEvent]): List[(String, Int)] // e.g. List(("nextPatTTL", 3050))
}

object PrintingStateKeeperDummy extends StateKeeper {
  override def handleEvent(ev: EricaEvent) = println("EricaEventLogger recovered " + ev)
  override def state(sampleTime: DateTime) = List()
}

object UselessFutureTellerDummy extends FutureTeller {
  override def futureState(events: List[EricaEvent]) = List() // not used
}

object WriteToCSV {
  var calledYet = false
  var filename = ""
  def apply(name: String, data: List[(String, Int)]) = {
    if(!calledYet) {
      calledYet = true
      filename = "output/" + name + DateTime.now() + ".csv"
      val csvWriter = CSVWriter.open(new File(filename))
      csvWriter.writeRow(data.map(_._1)) // fieldNames
      csvWriter.writeRow(data.map(_._2)) // values
      csvWriter.close()
    } else {
      val csvWriter = CSVWriter.open(new File(filename), append = true)
      csvWriter.writeRow(data.map(_._2))
      csvWriter.close()
    }
  }
}
