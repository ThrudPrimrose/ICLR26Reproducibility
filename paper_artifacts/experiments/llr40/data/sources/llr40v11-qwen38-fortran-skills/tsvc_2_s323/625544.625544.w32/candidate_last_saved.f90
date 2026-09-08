subroutine tsvc_2_s323_fp64(a, b, c, d, e, len_1d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d), b(len_1d)
  real(c_double), intent(in) :: c(len_1d), d(len_1d), e(len_1d)
  integer(c_int64_t) :: i, n
  real(c_double), allocatable :: b0(:), as(:), bs(:)
  real(c_double) :: mx_a, mx_b, err, ssum
  integer :: nthreads

  n = len_1d
  allocate(b0(n), as(n), bs(n))
  b0 = b

  ! --- my inscan version on the real arrays ---
  if (n >= 2) then
    ssum = 0.0d0
    !$omp parallel do simd reduction(inscan, +:ssum)
    do i = 2, n
      ssum = ssum + c(i) * (d(i) + e(i))
      !$omp scan inclusive(ssum)
      b(i) = b0(1) + ssum
      a(i) = b0(1) + ssum - c(i) * e(i)
    end do
    !$omp end parallel do simd
  end if

  ! --- serial reference on the copies ---
  as(1) = a(1); bs(1) = b(1)
  do i = 2, n
    as(i) = bs(i-1) + c(i) * d(i)
    bs(i) = as(i) + c(i) * e(i)
  end do

  ! --- compare ---
  mx_a = 0.0d0; mx_b = 0.0d0
  do i = 2, n
    err = abs(a(i) - as(i)); if (err > mx_a) mx_a = err
    err = abs(b(i) - bs(i)); if (err > mx_b) mx_b = err
  end do
  write(*,'(A,ES15.6)') 'MAX_ABS_ERR_A ', mx_a
  write(*,'(A,ES15.6)') 'MAX_ABS_ERR_B ', mx_b
  write(*,'(A,ES15.6)') 'REL_A ', mx_a/max(1.0d0,abs(as(n)))
  write(*,'(A,ES15.6)') 'REL_B ', mx_b/max(1.0d0,abs(bs(n)))
  write(*,'(A,I0)') 'NTHREADS=', omp_get_max_threads()
  nthreads = omp_get_max_threads()
  write(*,'(A,I0)') 'N=', n
  ! error at decimated points
  do i = 1000000, n, 5000000
    write(*,'(A,I0,A,ES12.4,A,ES12.4)') 'i=',i,' db=',b(i)-bs(i),' da=',a(i)-as(i)
  end do
  write(*,'(A,ES15.6,A,ES15.6)') 'END: db=', b(n)-bs(n), ' da=', a(n)-as(n)
  ! f stats and partial sums
  ssum = 0.0d0
  do i = 2, n
    ssum = ssum + c(i)*(d(i)+e(i))
  end do
  write(*,'(A,ES15.6)') 'SUM_F=', ssum
  write(*,'(A,ES15.6)') 'B1=', b0(1)
  write(*,'(A,ES15.6)') 'CMAX=', maxval(abs(c))
  write(*,'(A,ES15.6)') 'DMIN=', minval(d)
  write(*,'(A,ES15.6)') 'DMAX=', maxval(d)
  write(*,'(A,ES15.6)') 'EMIN=', minval(e)
  write(*,'(A,ES15.6)') 'EMAX=', maxval(e)
  flush(6)
end subroutine tsvc_2_s323_fp64
