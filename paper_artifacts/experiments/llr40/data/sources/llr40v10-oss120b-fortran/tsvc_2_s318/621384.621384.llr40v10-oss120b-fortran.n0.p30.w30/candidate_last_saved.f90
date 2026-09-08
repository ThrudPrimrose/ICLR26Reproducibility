! Fortran implementation of tsvc_2_s318_fp64 kernel
! Computes maximum absolute value of a double array with stride inc, and adds
! the index (0-based) where the max occurs, storing the result as a double.
!
! This version uses OpenMP parallelism to improve performance on multi-core.
! It follows the C reference implementation semantics exactly.
!
module tsvc_2_s318_mod
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(C, name="tsvc_2_s318_fp64")
    ! Arguments matching the C signature
    real(c_double), intent(in)  :: a(*)                ! input array (assumed-size)
    real(c_double), intent(out) :: result(1)           ! single-element output
    integer(c_int64_t), value   :: LEN_1D              ! number of elements to process
    integer(c_int64_t), value   :: inc                 ! stride (in elements)

    ! Local variables
    integer(c_int64_t) :: i, k, index
    real(c_double)    :: maxv, v, chksum
    integer           :: tid, nthreads
    real(c_double), allocatable :: local_max(:)
    integer(c_int64_t), allocatable :: local_idx(:)

    ! Initialise with the first element (i = 0)
    maxv = abs(a(1))
    index = 0_c_int64_t

    if (LEN_1D > 1_c_int64_t) then
      ! Determine the number of OpenMP threads and allocate per‑thread storage
      nthreads = omp_get_max_threads()
      allocate(local_max(nthreads))
      allocate(local_idx(nthreads))
      ! Initialise per‑thread locals to a value lower than any possible max
      local_max = -huge(0.0_c_double)
      local_idx = -1_c_int64_t

      !$omp parallel private(tid, i, k, v) shared(local_max, local_idx)
      tid = omp_get_thread_num() + 1  ! Fortran arrays are 1‑based

      !$omp do schedule(static) nowait
      do i = 1_c_int64_t, LEN_1D - 1_c_int64_t
        ! In the C reference k = i * inc (starting from i=1 gives inc)
        k = i * inc
        v = abs(a(k + 1))
        if (v > local_max(tid)) then
          local_max(tid) = v
          local_idx(tid) = i
        end if
      end do
      !$omp end do
      !$omp end parallel

      ! Reduce per‑thread results, preserving the smallest index on ties
      do tid = 1, nthreads
        if (local_max(tid) > maxv) then
          maxv = local_max(tid)
          index = local_idx(tid)
        else if (local_max(tid) == maxv) then
          if (local_idx(tid) >= 0_c_int64_t .and. local_idx(tid) < index) then
            index = local_idx(tid)
          end if
        end if
      end do

      deallocate(local_max)
      deallocate(local_idx)
    end if

    chksum = maxv + real(index, c_double)
    result(1) = chksum
  end subroutine tsvc_2_s318_fp64

end module tsvc_2_s318_mod
